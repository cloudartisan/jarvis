import cv2


def outline_rect(image, rect, colour):
    if rect is None:
        return
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x+w, y+h), colour)


def copy_rect(src, dst, src_rect, dst_rect, interpolation=cv2.INTER_LINEAR):
    """Copy part of the source to part of the destination."""
    x0, y0, w0, h0 = src_rect
    x1, y1, w1, h1 = dst_rect

    # Resize the contents of the source sub-rectangle.
    # Put the result in the destination sub-rectangle.
    dst[y1:y1+h1, x1:x1+w1] = cv2.resize(
        src[y0:y0+h0, x0:x0+w0], (w1, h1), interpolation=interpolation)


def swap_rects(src, dst, rects, interpolation=cv2.INTER_LINEAR):
    """Copy the source with two or more sub-rectangles swapped."""
    if dst is not src:
        dst[:] = src

    num_rects = len(rects)
    if num_rects < 2:
        return

    # Copy the contents of the last rectangle into temporary storage.
    x, y, w, h = rects[num_rects - 1]
    temp = src[y:y+h, x:x+w].copy()

    # Copy the contents of each rectangle into the next.
    i = num_rects - 2
    while i >= 0:
        copy_rect(src, dst, rects[i], rects[i+1], interpolation)
        i -= 1

    # Copy the temporarily stored content into the first rectangle.
    copy_rect(temp, dst, (0, 0, w, h), rects[0], interpolation)
