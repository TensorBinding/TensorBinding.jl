using Test

@testset "Fibonacci LDOS sampling planner" begin
    @testset "exact balanced intervals and samples" begin
        plan = TensorBinding.fibonacci_ldos_sampling_plan(
            4; num_x=3, num_avg=2,
        )

        @test plan.L == 4
        @test plan.N == 8
        @test plan.depth == 0
        @test plan.effective_L == 4
        @test plan.num_x == 3
        @test plan.num_x_requested == 3
        @test plan.num_avg == 2
        @test plan.intervals == [(-4):(-2), (-1):1, 2:3]
        @test plan.interval_rank_first == [0, 3, 6]
        @test plan.interval_rank_last == [2, 5, 7]
        @test plan.sample_ranks == [[0, 2], [3, 5], [6, 7]]
        @test plan.sample_conumbers == [[-4, -2], [-1, 1], [2, 3]]
        @test plan.conumber_axis == [-3, 0, 2]
        @test plan.groups == [[8, 6], [1, 7], [2, 5]]
        @test plan.centers == [3, 4, 2]
        @test plan.group_sizes == [2, 2, 2]
        @test plan.group_offsets == [1, 3, 5, 7]
        @test plan.group_offsets_zero == [0, 2, 4, 6]
        @test plan.group_offsets_base == 1
        @test plan.metadata.group_offsets_base == 1
        @test plan.sample_sites_flat == [8, 6, 1, 7, 2, 5]
        @test plan.sample_conumbers_flat == [-4, -2, -1, 1, 2, 3]
        @test plan.column_indices == [1, 2, 3]
        @test plan.total_samples == 6

        for column in plan.column_indices
            stored = plan.group_offsets[column]:(plan.group_offsets[column + 1] - 1)
            @test plan.sample_sites_flat[stored] == plan.groups[column]
            @test plan.sample_conumbers_flat[stored] == plan.sample_conumbers[column]

            # Python/HDF5 consumers use the zero-based half-open slice
            # flat[offsets[i]:offsets[i+1]]. Translate it by one for Julia.
            stored_zero = (plan.group_offsets_zero[column] + 1):plan.group_offsets_zero[column + 1]
            @test plan.sample_sites_flat[stored_zero] == plan.groups[column]
            @test plan.sample_conumbers_flat[stored_zero] ==
                  plan.sample_conumbers[column]
        end
    end

    @testset "inherited zoom coordinates" begin
        # L=10 -> L=4 after two atomic deflations. The original uncentered
        # ranks are 68:75, not a freshly assigned 0:7 reduced-chain axis.
        zoom = TensorBinding.fibonacci_ldos_sampling_plan(
            10; depth=2, num_x=3, num_avg=3, centered=false,
        )
        @test zoom.effective_L == 4
        @test zoom.window_count == 8
        @test zoom.window_first == 68
        @test zoom.window_last == 75
        @test zoom.intervals == [68:70, 71:73, 74:75]
        @test zoom.sample_conumbers == [[68, 69, 70], [71, 72, 73], [74, 75]]
        @test zoom.conumber_axis == [69, 72, 74]
        @test all(TensorBinding.fibonacci_conumber(
                      10, zoom.groups[column][sample]; centered=false,
                  ) == zoom.sample_conumbers[column][sample]
                  for column in eachindex(zoom.groups)
                  for sample in eachindex(zoom.groups[column]))

        # More requested columns/samples than sites produces singleton groups,
        # never duplicate samples or empty intervals.
        tiny = TensorBinding.fibonacci_ldos_sampling_plan(
            4; depth=0, num_x=100, num_avg=100,
        )
        @test tiny.num_x == 8
        @test tiny.group_sizes == ones(Int, 8)
        @test tiny.intervals == [conumber:conumber for conumber in (-4):3]
        @test tiny.groups == [[TensorBinding.fibonacci_site_from_conumber(4, c)]
                              for c in (-4):3]
    end

    @testset "conumber conventions" begin
        standard = TensorBinding.fibonacci_ldos_sampling_plan(
            8; depth=1, num_x=7, num_avg=4,
            orientation=:standard, alignment=:atomic,
            centered=true, origin=0,
        )
        reversed = TensorBinding.fibonacci_ldos_sampling_plan(
            8; depth=1, num_x=7, num_avg=4,
            orientation=:reversed, alignment=:atomic,
            centered=true, origin=0,
        )
        @test standard.intervals == reversed.intervals
        @test standard.sample_conumbers == reversed.sample_conumbers
        @test standard.groups != reversed.groups
        @test all(==(:atomic), TensorBinding.fibonacci_site_environment.(
                                  standard.L, standard.sample_sites_flat,
                              ))
        @test all(==(:atomic), TensorBinding.fibonacci_site_environment.(
                                  reversed.L, reversed.sample_sites_flat,
                              ))

        for plan in (standard, reversed)
            @test all(TensorBinding.fibonacci_conumber(
                          plan.L, plan.groups[column][sample];
                          orientation=plan.orientation,
                          alignment=plan.alignment,
                          centered=plan.centered,
                          origin=plan.origin,
                      ) == plan.sample_conumbers[column][sample]
                      for column in eachindex(plan.groups)
                      for sample in eachindex(plan.groups[column]))
        end

        # A noncanonical phase is meaningful for a complete depth-zero view,
        # but cannot be described as the nested atomic RG window.
        canonical_full = TensorBinding.fibonacci_ldos_sampling_plan(
            8; depth=0, num_x=7, num_avg=4,
        )
        raw = TensorBinding.fibonacci_ldos_sampling_plan(
            8; depth=0, num_x=7, num_avg=4, alignment=:raw,
        )
        shifted = TensorBinding.fibonacci_ldos_sampling_plan(
            8; depth=0, num_x=7, num_avg=4, origin=3,
        )
        @test canonical_full.intervals == raw.intervals == shifted.intervals
        @test canonical_full.groups != raw.groups
        @test canonical_full.groups != shifted.groups
        for plan in (raw, shifted)
            @test all(TensorBinding.fibonacci_conumber(
                          plan.L, plan.groups[column][sample];
                          orientation=plan.orientation,
                          alignment=plan.alignment,
                          centered=plan.centered,
                          origin=plan.origin,
                      ) == plan.sample_conumbers[column][sample]
                      for column in eachindex(plan.groups)
                      for sample in eachindex(plan.groups[column]))
        end

        @test_throws ArgumentError TensorBinding.fibonacci_ldos_sampling_plan(
            8; depth=1, alignment=:raw,
        )
        @test_throws ArgumentError TensorBinding.fibonacci_ldos_sampling_plan(
            8; depth=1, origin=1,
        )
    end

    @testset "L=43 remains output-sized" begin
        # Warm the exact specialization before measuring allocations.
        plan = TensorBinding.fibonacci_ldos_sampling_plan(
            43; depth=3, num_x=100, num_avg=5,
        )
        bytes = @allocated TensorBinding.fibonacci_ldos_sampling_plan(
            43; depth=3, num_x=100, num_avg=5,
        )

        @test plan.N == 1_134_903_170
        @test plan.effective_L == 34
        @test plan.num_x == 100
        @test length(plan.groups) == 100
        @test plan.total_samples == 500
        @test length(plan.sample_sites_flat) == 500
        @test all(length(group) == 5 for group in plan.groups)
        @test all(1 <= site <= plan.N for site in plan.sample_sites_flat)
        @test first(first(plan.intervals)) == plan.window_first
        @test last(last(plan.intervals)) == plan.window_last
        @test sum(length, plan.intervals) == plan.window_count
        @test all(last(plan.intervals[i]) + 1 == first(plan.intervals[i + 1])
                  for i in 1:(plan.num_x - 1))
        @test bytes < 20_000_000

        deepest_requested_view = TensorBinding.fibonacci_ldos_sampling_plan(
            43; depth=12, num_x=100, num_avg=5,
        )
        @test deepest_requested_view.effective_L == 7
        @test deepest_requested_view.window_count == 34
        @test deepest_requested_view.num_x == 34
        @test deepest_requested_view.group_sizes == ones(Int, 34)
    end

    @testset "Hamiltonian overload and validation" begin
        Hfib = TensorBinding.fibonacci_hamiltonian(
            2; A=1.0, B=2.0, boundary=:open,
        )
        from_H = TensorBinding.fibonacci_ldos_sampling_plan(
            Hfib; num_x=2, num_avg=1,
        )
        from_L = TensorBinding.fibonacci_ldos_sampling_plan(
            2; num_x=2, num_avg=1,
        )
        @test from_H.groups == from_L.groups
        @test from_H.metadata == from_L.metadata

        Hbinary = TensorBinding.get_Hamiltonian("chain_1d", 1.0; L=2)
        @test_throws ArgumentError TensorBinding.fibonacci_ldos_sampling_plan(Hbinary)
        @test_throws ArgumentError TensorBinding.fibonacci_ldos_sampling_plan(4; num_x=0)
        @test_throws ArgumentError TensorBinding.fibonacci_ldos_sampling_plan(4; num_avg=0)
        @test_throws ArgumentError TensorBinding.fibonacci_ldos_sampling_plan(
            4; orientation=:sideways,
        )
        @test_throws ArgumentError TensorBinding.fibonacci_ldos_sampling_plan(
            4; alignment=:molecular,
        )
        @test_throws ArgumentError TensorBinding.fibonacci_ldos_sampling_plan(
            4; depth=1,
        )
    end
end
