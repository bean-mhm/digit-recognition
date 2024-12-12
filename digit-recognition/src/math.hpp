
#pragma once

#include <algorithm>
#include <cmath>

namespace math
{

    // linear interpolation
    static float mix(float a, float b, float t)
    {
        return a + t * (b - a);
    }

    // Euclidean distance of a 2D point on the cartesian plane from the origin
    static float length(float x, float y)
    {
        return std::sqrt(x * x + y * y);
    }

    // distance of a point and a line segment
    static float dist_segment(
        float px, float py, // point coordinates
        float ax, float ay, // starting point of line segment
        float bx, float by // end point of line segment
    )
    {
        float ap_x = px - ax;
        float ap_y = py - ay;

        float ab_x = bx - ax;
        float ab_y = by - ay;

        float dot_ap_ab = (ap_x * ab_x) + (ap_y * ab_y);
        float dot_ab_ab = (ab_x * ab_x) + (ab_y * ab_y);

        float h = std::clamp(dot_ap_ab / dot_ab_ab, 0.f, 1.f);
        return length(ap_x - ab_x * h, ap_y - ab_y * h);
    }

    // map a number from the [start, end] range to the [0, 1] range with
    // clamping.
    static float remap01(float x, float start, float end)
    {
        return std::clamp((x - start) / (end - start), 0.f, 1.f);
    }

    // map a number from the [from_start, from_end] range to the
    // [to_start, to_end] range.
    static float remap(
        float x,
        float from_start,
        float from_end,
        float to_start,
        float to_end
    )
    {
        float t = (x - from_start) / (from_end - from_start);
        return to_start + t * (to_end - to_start);
    }

    // map a number from the [from_start, from_end] range to the
    // [to_start, to_end] range with clamping.
    static float remap_clamp(
        float x,
        float from_start,
        float from_end,
        float to_start,
        float to_end
    )
    {
        float t =
            std::clamp((x - from_start) / (from_end - from_start), 0.f, 1.f);
        return to_start + t * (to_end - to_start);
    }

}
