# Figure generation for benchmark verification. Included by scripts only when
# --plot is given (CairoMakie load is slow). Figures follow the panel layout
# of the source papers so the flow structure (rotors, arch, front) can be
# compared directly against the published solutions.

using CairoMakie

"""
    save_benchmark_figure(path, x, z, panels; title="")

Save stacked contour panels at the final output time. `panels` is a vector of
`(field, label, levels)` tuples where `field` is `(kDim, ncols)` with z
varying fastest. Negative contours are dashed in the line overlay, matching
the conventions of the benchmark papers.
"""
function save_benchmark_figure(path::String, x::AbstractVector, z::AbstractVector,
                               panels::Vector; title::String="")
    fig = Figure(size = (950, 80 + 320 * length(panels)))
    for (i, (field, label, levels)) in enumerate(panels)
        ax = Axis(fig[i, 1],
                  title = i == 1 ? title : "",
                  xlabel = i == length(panels) ? "x (km)" : "",
                  ylabel = "z (km)")
        cf = contourf!(ax, x ./ 1000.0, z ./ 1000.0, field',
                       levels = levels, extendlow = :auto, extendhigh = :auto)
        contour!(ax, x ./ 1000.0, z ./ 1000.0, field',
                 levels = levels, color = :black, linewidth = 0.5)
        Colorbar(fig[i, 2], cf, label = label)
    end
    save(path, fig)
    println("Saved figure: $path")
    return path
end
