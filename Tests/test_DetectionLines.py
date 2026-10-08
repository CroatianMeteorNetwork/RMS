""" Line finding of the meteor detection: the threshold passers of the time windows with the frame in which
they peaked, the 3D line search, the selection of the 3D line belonging to a 2D line, the joining of the
pieces of a track, and the removal of duplicates. """

import numpy as np

import RMS.ConfigReader as cr
from RMS.Detection import extendPieces, joinContinuousDetections, merge3DLines, removeDuplicateDetections, \
    stitchedLineDeviation
from RMS.DetectionTools import getWindowStripePoints, windowPoints, WindowPoints, \
    windowThresholdConnected
from RMS.Routines.DynamicFTPCompressionCy import FFMimickInterface
from RMS.Routines.LineFinder3D import findLines3D, selectSeedLine, stitch3DLines
from RMS.Routines.SequentialRANSAC import getPolarLine


def _config():
    config = cr.Config()
    config.stripe_width = 20
    return config


def test_maxframe_is_the_frame_of_the_maximum():

    frames = np.random.default_rng(0).integers(0, 1000, (40, 6, 7)).astype(np.uint16)

    ff = FFMimickInterface(6, 7, np.uint16)
    for frame in frames:
        ff.addFrame(frame)
    ff.finish()

    # The first frame of the maximum, as np.argmax
    assert np.array_equal(ff.maxframe, np.argmax(frames, axis=0))
    assert np.array_equal(ff.maxpixel, frames.max(axis=0))


def _movingObjectWindows(n_frames=1024, window=512, slide=256, x0=50.0, y0=100.0, vx=0.2, vy=0.05):
    """ Window points of an object moving 1 px every 5 frames, as the line search makes them. """

    window_points = []
    for frame_min in range(0, n_frames - window + 1, slide):

        ff = FFMimickInterface(200, 300, np.uint16)
        for fr in range(frame_min, frame_min + window + 1):
            # The object is 3x3 px, so that the lonely pixel removal keeps its track
            img = np.full((200, 300), 100, dtype=np.uint16)
            x, y = int(round(x0 + vx*fr)), int(round(y0 + vy*fr))
            img[y - 1:y + 2, x - 1:x + 2] = 1000
            ff.addFrame(img)
        ff.finish()

        img_thresh = (ff.maxpixel.astype(np.int32) - ff.avepixel > 100).astype(np.uint8)

        window_points.append(windowPoints(ff, img_thresh, frame_min, frame_min + window))

    return window_points


def test_window_stripe_points_follow_the_object():

    config = _config()
    window_points = _movingObjectWindows()

    # Line through the track, (50, 100) to (254.8, 151.2)
    x1, y1, x2, y2 = 50.0, 100.0, 50.0 + 0.2*1024, 100.0 + 0.05*1024
    rho, theta = getPolarLine(x1, y1, x2, y2, 300, 200)

    xs, ys, zs = getWindowStripePoints(config, window_points, 0, 1024, rho, theta, 200, 300,
        line_start=(x1, y1), line_end=(x2, y2))

    assert len(xs) > 100

    # The points have the frames in which the object was there
    assert np.all(np.abs(xs - (50 + 0.2*zs)) <= 2.5)
    assert np.all(np.abs(ys - (100 + 0.05*zs)) <= 2.5)

    # Each pixel and frame is taken once, although the windows overlap
    assert len(np.unique(np.column_stack([xs, ys, zs]), axis=0)) == len(xs)

    # Only the points in the frame range are taken
    xs_part, _, zs_part = getWindowStripePoints(config, window_points, 300, 400, rho, theta, 200, 300,
        line_start=(x1, y1), line_end=(x2, y2))
    assert len(xs_part) and (zs_part.min() >= 300) and (zs_part.max() <= 400)

    # Points outside the stripe are not taken
    rho_far, theta_far = getPolarLine(x1, y1 + 60, x2, y2 + 60, 300, 200)
    xs_far, _, _ = getWindowStripePoints(config, window_points, 0, 1024, rho_far, theta_far, 200, 300,
        line_start=(x1, y1 + 60), line_end=(x2, y2 + 60))
    assert len(xs_far) == 0


def _trackPoints(rng, x0, y0, vx, vy, f0, f1, noise=0.5):
    fr = np.arange(f0, f1, dtype=np.float64)
    return np.column_stack([x0 + vx*(fr - f0) + rng.normal(0, noise, len(fr)),
                            y0 + vy*(fr - f0) + rng.normal(0, noise, len(fr)), fr])


def test_3d_line_search_is_reproducible():

    rng = np.random.default_rng(3)
    points = np.vstack([_trackPoints(rng, 50, 50, 0.5, 0.2, 0, 200),
                        np.column_stack([rng.uniform(0, 300, 300), rng.uniform(0, 300, 300),
                                         rng.uniform(0, 200, 300)])])

    kwargs = dict(max_lines=5, min_points=20, dist_thresh=4.0, max_gap_frame=50, max_gap_spatial=50,
                  min_frames=10, img_w=300, img_h=300, max_iterations=100)

    # The global random state doesn't change the result
    np.random.seed(1)
    lines1 = findLines3D(points, **kwargs)
    np.random.seed(2)
    lines2 = findLines3D(points, **kwargs)

    assert len(lines1) == len(lines2) >= 1
    for l1, l2 in zip(lines1, lines2):
        assert np.allclose(l1[0], l2[0]) and np.allclose(l1[1], l2[1])


def _line(x1, y1, f1, x2, y2, f2, n=100):
    return [(x1, y1, f1), (x2, y2, f2), n, 1.0, int(f1), int(f2)]


def test_seed_line_is_the_one_along_the_2d_line():

    # 2D line from (100, 100) to (300, 200)
    ref = (100.0, 100.0, 300.0, 200.0)

    along_short = _line(100, 100, 0, 150, 125, 50)
    along_long = _line(150, 125, 50, 300, 200, 200)

    # Another object crossing the stripe, its middle is close to the middle of the 2D line
    crossing = _line(160, 120, 300, 240, 160 + 0.7*80, 380)

    # Another object moving side by side, 8 px away
    side = _line(100 - 3.6, 100 + 7.2, 0, 300 - 3.6, 200 + 7.2, 400)

    assert selectSeedLine([along_short, crossing, along_long], ref) is along_long
    assert selectSeedLine([side, along_short], ref) is along_short
    assert selectSeedLine([crossing], ref) is None

    # The stitching starts from the line along the 2D line and joins the other part of the object
    stitched = stitch3DLines([crossing, along_short, along_long], ref, ref_has_frames=False, dist_thresh=5.0,
        frame_scale=1.0)
    assert (stitched[4], stitched[5]) == (0, 200)


def test_parallel_objects_are_not_merged():

    # Two objects moving side by side, 8 px apart, and a duplicate of the first one
    first = _line(100, 100, 0, 300, 200, 400)
    duplicate = _line(105.5, 102.5, 10, 295.5, 197.5, 390)
    side = _line(100 - 3.6, 100 + 7.2, 0, 300 - 3.6, 200 + 7.2, 400)

    merged = merge3DLines([first, duplicate, side], 512, 512, dist_thresh=4.0)
    assert len(merged) == 2

    # The wider default limit merges them all
    assert len(merge3DLines([first, duplicate, side], 512, 512)) == 1


def test_duplicate_detections_are_removed():

    frames = np.arange(100, 200, dtype=np.float64)

    # The object moves by (0.5, 0.1) px per frame, the unit vector across the motion is perp
    perp = np.array([-0.1, 0.5])/np.hypot(0.1, 0.5)
    along = np.array([0.5, 0.1])/np.hypot(0.1, 0.5)

    def detection(frames, offset):
        return [0.0, 0.0, np.column_stack([frames, 50 + frames*0.5 + offset[0], 80 + frames*0.1 + offset[1],
                                           np.ones(len(frames))])]

    long_det = detection(frames, (0, 0))

    # A shorter copy 1 px away, and a copy shifted along the track by 6 px (as the centroids of an extended
    #   object found around two lines can be)
    duplicate = detection(frames[10:60], perp*1.0)
    shifted = detection(frames[5:95], along*6.0)

    # Another object moving side by side 8 px away, and one later along the same line
    side = detection(frames, perp*8.0)
    other_time = detection(frames + 300, (0, 0))

    kept = removeDuplicateDetections([duplicate, long_det, side, other_time, shifted], 4.0, 20.0)

    assert len(kept) == 3
    assert all((det is not duplicate) and (det is not shifted) for det in kept)


def test_pieces_of_a_track_are_joined():

    def detection(f0, f1, x0, y0, vx, vy, ax=0.0):
        fr = np.arange(f0, f1, dtype=np.float64)
        t = fr - f0
        cent = np.column_stack([fr, x0 + vx*t + 0.5*ax*t**2, y0 + vy*t, np.ones(len(fr))])
        return [0.0, 0.0, cent]

    # An accelerating object in two pieces with a gap of 10 frames, the second piece begins where the motion
    #   of the first one leads
    first = detection(0, 200, 100, 100, 2.0, 1.0, ax=0.002)
    x_end, vx_end = 100 + 2.0*209 + 0.001*209**2, 2.0 + 0.002*209
    second = detection(210, 400, x_end, 100 + 210, vx_end, 1.0)

    # Another object which begins later along the same line, but at a different position
    other = detection(220, 300, x_end + 60, 100 + 230, vx_end, 1.0)

    joined = joinContinuousDetections([second, other, first], 50, 50, 5.0, 1080, 1920)

    assert len(joined) == 2
    frames = joined[0][2][:, 0]
    assert (frames[0], frames[-1]) == (0, 399)


def test_stitched_line_deviation():

    line = [(0, 0, 0), (100, 0, 100)]
    straight = [_line(0, 0, 0, 50, 0, 50), _line(50, 0, 50, 100, 0, 100)]
    bent = [_line(0, 0, 0, 50, 10, 50), _line(50, 10, 50, 100, 0, 100)]

    assert stitchedLineDeviation(line, straight) == 0
    assert abs(stitchedLineDeviation(line, bent) - 10) < 1e-9


def test_pieces_are_extended_over_the_gaps():

    pieces = [_line(0, 0, 0, 100, 0, 100), _line(140, 0, 140, 200, 0, 200)]

    extended = extendPieces(pieces)

    # The gap 100 - 140 is split in the middle, each piece continues along its own motion
    assert (extended[0][4], extended[0][5]) == (0, 120)
    assert (extended[1][4], extended[1][5]) == (120, 200)
    assert np.allclose(extended[0][1], (120, 0, 120))
    assert np.allclose(extended[1][0], (120, 0, 120))


def test_overlapping_pieces_are_joined():

    def detection(f0, f1):
        fr = np.arange(f0, f1, dtype=np.float64)
        return [0.0, 0.0, np.column_stack([fr, 100 + 0.1*fr, 200 + 0.05*fr, np.ones(len(fr))])]

    # A slow object in two pieces which overlap by 100 frames
    joined = joinContinuousDetections([detection(300, 1000), detection(0, 400)], 50, 50, 4.0, 512, 512)

    assert len(joined) == 1
    assert (joined[0][2][0, 0], joined[0][2][-1, 0]) == (0, 999)

    # A slow object in two pieces with a gap of 250 frames, in which it moves by 28 px
    joined = joinContinuousDetections([detection(650, 1000), detection(0, 400)], 50, 50, 4.0, 512, 512)

    assert len(joined) == 1
    assert (joined[0][2][0, 0], joined[0][2][-1, 0]) == (0, 999)


def test_extended_object_duplicates_are_removed():

    frames = np.arange(100, 200, dtype=np.float64)

    # A wide object moving horizontally along y = 50 (lit between y = 40 and 60), and an object moving side by
    #   side along y = 75
    img = np.zeros((100, 400), dtype=np.uint8)
    img[40:61, :] = 1
    img[73:78, :] = 1
    window_points = [WindowPoints(0, 512, None, None, None, np.packbits(img > 0), img.shape)]

    def connected(frame, x1, y1, x2, y2):
        return windowThresholdConnected(window_points, frame, x1, y1, x2, y2)

    def detection(y, frames=frames):
        return [0.0, 0.0, np.column_stack([frames, 10 + frames, np.full(len(frames), y),
                                           np.ones(len(frames))])]

    # The detection at the centre is the longest one, so it is kept
    centre = detection(50, np.arange(90, 210, dtype=np.float64))

    # Two detections of the wide object 7 px from its centre, and the other object 25 px away
    edge_low, edge_high = detection(43), detection(57)
    side = detection(75)

    kept = removeDuplicateDetections([edge_low, centre, edge_high, side], 4.0, 20.0, connected=connected,
        max_connected_dist=30.0)
    assert [det[2][0, 2] for det in kept] == [50, 75]

    # Without the thresholded images, the edges are kept
    assert len(removeDuplicateDetections([edge_low, centre, edge_high, side], 4.0, 20.0)) == 4


def test_pieces_of_a_binned_track_are_joined():

    # The pieces of a slow track, with the centroids in the image binned 2x2 or scaled to the unbinned image
    #   (the distances and the image size stay in the binned image)
    def detection(f0, f1, bin_factor):
        fr = np.arange(f0, f1, dtype=np.float64)
        x, y = 100 + 0.1*fr, 200 + 0.05*fr
        return [0.0, 0.0, np.column_stack([fr, bin_factor*x + (bin_factor - 1)/2.0,
                                           bin_factor*y + (bin_factor - 1)/2.0, np.ones(len(fr))])]

    binned = joinContinuousDetections([detection(650, 1000, 2), detection(0, 400, 2)], 50, 50, 4.0, 512, 512,
        bin_factor=2)
    unbinned = joinContinuousDetections([detection(650, 1000, 1), detection(0, 400, 1)], 50, 50, 4.0, 512,
        512)

    assert len(binned) == len(unbinned) == 1

    # The polar line is in the binned image, the same as without binning
    assert np.allclose(binned[0][:2], unbinned[0][:2])
