@testset "models" begin

    # ── LineModel ─────────────────────────────────────────────────────────────

    @testset "LineModel: point_distance" begin
        # Line x=0 (vertical), A=1, B=0, C=0
        m = LineModel(1.0, 0.0, 0.0)
        @test point_distance(m, (0.0, 5.0))  ≈ 0.0 atol=1e-12
        @test point_distance(m, (3.0, 5.0))  ≈ 3.0 atol=1e-12
        @test point_distance(m, (-2.0, 0.0)) ≈ 2.0 atol=1e-12
    end

    @testset "LineModel: point on oblique line has zero distance" begin
        pts = [(x, 2.0*x + 1.0) for x in 1.0:5.0]
        A, B, C = fit_line_tls(pts)
        m = LineModel(A, B, C)
        for p in pts
            @test point_distance(m, p) < 1e-9
        end
    end

    @testset "LineModel: constraints_met — angle within range" begin
        # Horizontal line (angle=0): should pass default constraints (min=-π/2, max=π/2)
        m_horiz = LineModel(0.0, 1.0, -5.0)   # B*y = 5, horizontal
        c = LineConstraints{Float64}()
        @test constraints_met(m_horiz, c)
    end

    @testset "LineModel: constraints_met — angle outside range" begin
        c = LineConstraints{Float64}(min_angle=Float64(π/4), max_angle=Float64(π/2))
        # Horizontal line angle ≈ 0, outside [π/4, π/2]
        m_horiz = LineModel(0.0, 1.0, 0.0)
        @test !constraints_met(m_horiz, c)
    end

    @testset "LineModel: rms_error is zero for exact points" begin
        pts = [(x, 3.0) for x in 1.0:5.0]
        A, B, C = fit_line_tls(pts)
        m = LineModel(A, B, C)
        @test rms_error(m, pts) < 1e-9
    end

    # ── LineSegmentModel ──────────────────────────────────────────────────────

    @testset "LineSegmentModel: segment_length" begin
        # Points along y=0, x from 1 to 10 → segment of length ~9
        pts = [(Float64(x), 0.0) for x in 1:10]
        A, B, C = fit_line_tls(pts)
        m = LineSegmentModel(A, B, C)
        @test segment_length(m, pts) ≈ 9.0 atol=0.01
    end

    @testset "LineSegmentModel: empty inlier set returns 0" begin
        m = LineSegmentModel(1.0, 0.0, 0.0)
        @test segment_length(m, []) == 0.0
    end

    # ── CircleModel ───────────────────────────────────────────────────────────

    @testset "CircleModel: point_distance — point on circle" begin
        m = CircleModel(0.0, 0.0, 10.0)
        pt_on = (10.0, 0.0)
        @test point_distance(m, pt_on) ≈ 0.0 atol=1e-12
    end

    @testset "CircleModel: point_distance — interior and exterior" begin
        m = CircleModel(0.0, 0.0, 10.0)
        @test point_distance(m, (5.0, 0.0))  ≈ 5.0  atol=1e-10  # inside: 10-5
        @test point_distance(m, (15.0, 0.0)) ≈ 5.0  atol=1e-10  # outside: 15-10
    end

    @testset "CircleModel: constraints_met — radius in range" begin
        m = CircleModel(0.0, 0.0, 25.0)
        @test  constraints_met(m, CircleConstraints{Float64}(min_radius=10.0, max_radius=50.0))
        @test !constraints_met(m, CircleConstraints{Float64}(min_radius=30.0, max_radius=50.0))
        @test !constraints_met(m, CircleConstraints{Float64}(min_radius=1.0,  max_radius=20.0))
    end

    @testset "CircleModel: arc_completeness — full circle" begin
        m = CircleModel(0.0, 0.0, 10.0)
        pts = [(10.0*cos(θ), 10.0*sin(θ)) for θ in range(0, 2π, length=37)[1:end-1]]
        @test arc_completeness(m, pts) ≈ 1.0 atol=0.01
    end

    @testset "CircleModel: arc_completeness — semicircle ≈ 0.5" begin
        m = CircleModel(0.0, 0.0, 10.0)
        pts = [(10.0*cos(θ), 10.0*sin(θ)) for θ in range(0, π, length=19)[1:end-1]]
        comp = arc_completeness(m, pts)
        @test 0.4 <= comp <= 0.6
    end

    @testset "CircleModel: arc_completeness — empty returns 0" begin
        m = CircleModel(0.0, 0.0, 10.0)
        @test arc_completeness(m, []) == 0.0
    end

    # ── sample_size interface ─────────────────────────────────────────────────

    @testset "sample_size matches DOF" begin
        @test sample_size(LineModel)        == 2
        @test sample_size(LineSegmentModel) == 2
        @test sample_size(CircleModel)      == 3
    end

    # ── rms_error ─────────────────────────────────────────────────────────────

    @testset "rms_error on exact circle is zero" begin
        m = CircleModel(0.0, 0.0, 5.0)
        pts = [(5.0*cos(θ), 5.0*sin(θ)) for θ in range(0, 2π, length=20)[1:end-1]]
        @test rms_error(m, pts) < 1e-10
    end

    @testset "rms_error on empty set is zero" begin
        @test rms_error(LineModel(1.0, 0.0, 0.0), []) == 0.0
        @test rms_error(CircleModel(0.0, 0.0, 1.0), []) == 0.0
    end

    # ── Parametric element type ───────────────────────────────────────────────

    @testset "parametric models preserve element type (Float32)" begin
        lm = LineModel{Float32}(0.0f0, 1.0f0, -5.0f0)
        @test lm isa LineModel{Float32}
        @test typeof(lm.A) === Float32
        @test point_distance(lm, (3.0f0, 5.0f0)) === 0.0f0

        cm = CircleModel{Float32}(0.0f0, 0.0f0, 10.0f0)
        @test cm isa CircleModel{Float32}
        @test typeof(point_distance(cm, (10.0f0, 0.0f0))) === Float32

        # Float64 mixed-type construction still works via promotion
        @test LineModel(1, 2, 3) isa LineModel{Float64}
    end

    @testset "parametric fit_model routes by type parameter" begin
        pts64 = [(x, 2x + 1.0) for x in 1.0:5.0]
        m64 = fit_model(LineModel{Float64}, pts64)
        @test m64 isa LineModel{Float64}

        pts32 = [(Float32(x), 2f0*Float32(x) + 1f0) for x in 1:5]
        m32 = fit_model(LineModel{Float32}, pts32)
        @test m32 isa LineModel{Float32}
        @test typeof(m32.A) === Float32
    end

    # ── data_constraints_met (Step 13) ────────────────────────────────────────

    @testset "data_constraints_met: generic fallback is true" begin
        lm = LineModel(0.0, 1.0, -5.0)
        @test data_constraints_met(lm, LineConstraints{Float64}(), [(0.0, 5.0), (1.0, 5.0)])
    end

    @testset "data_constraints_met: CircleModel enforces arc completeness" begin
        m = CircleModel(0.0, 0.0, 10.0)
        c_loose  = CircleConstraints{Float64}(min_completeness=0.1)
        c_strict = CircleConstraints{Float64}(min_completeness=0.9)
        full_pts = [(10.0*cos(θ), 10.0*sin(θ)) for θ in range(0, 2π, length=37)[1:end-1]]
        half_pts = [(10.0*cos(θ), 10.0*sin(θ)) for θ in range(0, π, length=19)[1:end-1]]
        @test  data_constraints_met(m, c_loose,  full_pts)
        @test  data_constraints_met(m, c_strict, full_pts)
        @test  data_constraints_met(m, c_loose,  half_pts)
        @test !data_constraints_met(m, c_strict, half_pts)
        @test !data_constraints_met(m, c_strict, [])
    end

end
