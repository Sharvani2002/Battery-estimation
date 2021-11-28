function [dx] = clip(dx, max_clip_value, min_clip_value)

if max(dx) > max_clip_value
    dx(dx > max_clip_value) = max_clip_value;
end

if min(dx) < min_clip_value
    dx(dx < min_clip_value) = min_clip_value;
end
end
            