""" In-app help content for SkyFit2.

This module holds the user-facing documentation shown in the SkyFit2 "Help" tab. It is kept
separate from the GUI code so the text is easy to edit. Content is rendered as the rich-text HTML
subset understood by QTextBrowser (headings, paragraphs, lists, tables, links) - do NOT rely on
Markdown, as setMarkdown() requires Qt >= 5.14.

The Help tab uses progressive disclosure: buildHelpHome() returns a short intro and a "what do you
want to do?" triage list whose links open detailed topic pages built by buildHelpTopic(). Only the
topics relevant to the current mode ('skyfit' or 'manualreduction') and the enabled features
(geopoints, satellite tracks, FR files, DFN/Debruijn, ...) are shown.

Source material: Guides/SkyFit.md, the on-image keyboard hints previously in Utils/SkyFit2.py, and
the GMN wiki (see WIKI_URL).
"""

from __future__ import absolute_import, division, print_function


# Full online manual
WIKI_URL = "https://globalmeteornetwork.org/wiki/index.php?title=SkyFit2"


# Stylesheet applied to the QTextBrowser document (the Qt rich-text CSS subset: fonts, colours,
# block margins, table cell padding). Keeps the help readable: clear heading hierarchy, breathing
# room between blocks, soft "keycap" styling for shortcuts, and muted descriptions.
HELP_STYLE = """
    body { font-family: "Segoe UI", "DejaVu Sans", sans-serif; font-size: 10pt; color: #202124; }
    h2 { font-size: 15pt; font-weight: bold; color: #16324f; margin-top: 0px; margin-bottom: 10px; }
    h3 { font-size: 11pt; font-weight: bold; color: #16324f; margin-top: 16px; margin-bottom: 4px; }
    p { margin-top: 4px; margin-bottom: 10px; }
    p.lead { color: #3c4043; font-size: 10.5pt; }
    ol, ul { margin-top: 2px; margin-bottom: 10px; }
    li { margin-bottom: 6px; }
    a { color: #1a5fb4; text-decoration: none; font-weight: bold; }
    td { padding-top: 4px; padding-bottom: 4px; padding-right: 12px; }
    td.desc { color: #5f6368; font-weight: normal; }
    span.key { font-family: "Consolas", "DejaVu Sans Mono", monospace; background-color: #e8eaed;
               color: #16324f; font-weight: bold; }
    span.btn { background-color: #dfe6ef; color: #16324f; font-weight: bold; }
    span.tip { color: #5f6368; }
    p.navnext { margin-top: 8px; margin-bottom: 2px; }
    p.navrel { margin-top: 2px; margin-bottom: 2px; color: #5f6368; }
"""


# ---------------------------------------------------------------------------------------------- #
#  Small helpers for reading GUI state safely (the Help tab must never crash the app).
# ---------------------------------------------------------------------------------------------- #

def _mode(gui):
    return getattr(gui, 'mode', 'skyfit')


def _ctrl(gui):
    """ Modifier label ("CTRL" or "CMD" on macOS). """
    return getattr(gui, 'ctrl_label', 'CTRL')


def _input_type(gui):
    """ Best-effort read of the current input type ('ff', 'video', 'images', 'dfn', ...). """
    for path in (lambda g: g.img.img_handle.input_type, lambda g: g.img_handle.input_type):
        try:
            return path(gui)
        except Exception:
            continue
    return ''


def _has_geopoints(gui):
    return getattr(gui, 'geo_points_obj', None) is not None


def _has_sattracks(gui):
    return bool(getattr(gui, 'show_sattracks', False)) or (getattr(gui, 'tle_file', None) is not None)


def _has_fr(gui):
    return bool(getattr(gui, 'use_fr_files', False))


def _is_dfn(gui):
    return _input_type(gui) == 'dfn'


# ---------------------------------------------------------------------------------------------- #
#  HTML formatting helpers.
# ---------------------------------------------------------------------------------------------- #

def _key(text):
    """ Render a keyboard key/shortcut as a soft keycap. """
    return '<span class="key">&nbsp;{:s}&nbsp;</span>'.format(text)


def _shortcut_table(rows):
    """ Build a two-column shortcut table from a list of (keys, description) tuples. """
    html = '<table cellspacing="0" cellpadding="0">'
    for keys, desc in rows:
        html += ('<tr><td valign="top">{k}</td>'
                 '<td valign="top" class="desc">{d}</td></tr>').format(k=_key(keys), d=desc)
    html += '</table>'
    return html


def _defn_table(rows):
    """ Build a two-column "term -> description" table from (term_html, description) tuples. """
    html = '<table cellspacing="0" cellpadding="0">'
    for term, desc in rows:
        html += ('<tr><td valign="top"><b>{t}</b></td>'
                 '<td valign="top" class="desc">{d}</td></tr>').format(t=term, d=desc)
    html += '</table>'
    return html


def _btn(text):
    """ Render a reference to a clickable UI button (distinct from a keyboard key). """
    return '<span class="btn">&nbsp;{:s}&nbsp;</span>'.format(text)


def _callout(html, kind="tip"):
    """ Render a tip/note callout box. kind: 'tip' (blue) or 'note' (amber). """
    if kind == "note":
        bg, fg, label = "#fff4e5", "#9a5b00", "Note"
    else:
        bg, fg, label = "#e8f0fe", "#1a5fb4", "Tip"
    return ('<table cellspacing="0" cellpadding="7" width="100%"><tr>'
            '<td style="background-color:{bg};"><b style="color:{fg};">{label}:</b> {html}</td>'
            '</tr></table>').format(bg=bg, fg=fg, label=label, html=html)


def _nav_links(next_pair=None, related=None):
    """ Footer with a "Next" link and/or a list of "Related" links, each on its own line.

    next_pair: (topic_id, title) or None.  related: list of (topic_id, title) or None.
    """
    html = ""
    if next_pair:
        html += ('<p class="navnext"><b>Next:</b> '
                 '<a href="topic:{0}">{1} &rarr;</a></p>').format(next_pair[0], next_pair[1])
    if related:
        links = ' &nbsp;&middot;&nbsp; '.join(
            '<a href="topic:{0}">{1}</a>'.format(t, n) for t, n in related)
        html += '<p class="navrel"><b>Related:</b> ' + links + '</p>'
    if not html:
        return ""
    return '<hr>' + html


def _page(title, body_html):
    """ Wrap a topic body with a consistent heading. """
    return '<h2>{:s}</h2>{:s}'.format(title, body_html)


# ---------------------------------------------------------------------------------------------- #
#  Topic registry.
#
#  Each entry: id -> dict(title, modes, enabled, build)
#    - modes:   tuple of modes the topic appears in ('skyfit', 'manualreduction')
#    - enabled: callable(gui) -> bool, controls whether the topic is shown (feature gating)
#    - build:   callable(gui) -> str, the topic detail HTML
# ---------------------------------------------------------------------------------------------- #

def _always(gui):
    return True


# ----- SkyFit topics -------------------------------------------------------------------------- #

def _topic_overview(gui):
    c = _ctrl(gui)
    body = (
        "<p>SkyFit calibrates an image against the star catalog so pixel positions can be turned "
        "into sky coordinates (right ascension / declination), and pixel brightness into "
        "magnitudes. You produce a <b>platepar</b> (short for <i>plate parameters</i>) that is then "
        "used to measure meteors.</p>"

        "<h3>Auto mode (recommended)</h3>"
        "<p>Quick and easy &ndash; SkyFit does the solving for you.</p>"
        "<ol>"
        "<li>When SkyFit opens, pick an existing platepar to refine, or close the dialog to start "
        "fresh.</li>"
        "<li>Set the image brightness in the <b>Levels</b> tab on the right (drag the histogram "
        "handles), or press " + _key(c + " + A") + " for auto levels.</li>"
        "<li>Open the <b>Fit Parameters</b> tab and click " + _btn("Auto Fit") + ". It automatically "
        "detects stars and solves the plate (pointing <i>and</i> distortion), using astrometry.net "
        "if needed &ndash; no manual star picking required. Photometry is fitted automatically too.</li>"
        "<li>Check the fit: open the two residual plots with the " + _btn("Astrometry") + " and "
        + _btn("Photometry") + " buttons and make sure they look good &ndash; see "
        "<a href=\"topic:residuals\">checking the fit</a> for the targets to hit.</li>"
        "<li>Save with " + _key(c + " + S") + ".</li>"
        "</ol>"
        + _callout(_btn("Auto Pointing") + " re-solves only the pointing while keeping the existing "
        "distortion &ndash; handy when re-calibrating a known lens.")

        + "<h3>Manual mode (when you need it)</h3>"
        "<p>Use this if auto solving struggles (few stars, very wrong starting point) or you want "
        "full control.</p>"
        "<ol>"
        "<li>Roughly line up the catalog (cyan) with the image stars using "
        + _key("A/D") + " (azimuth), " + _key("S/W") + " (altitude), " + _key("Q/E") + " (rotation), "
        + _key("Up/Down") + " (scale). Get them <i>close enough</i>, not exact.</li>"
        "<li>Press " + _key(c + " + R") + " to enter star-picking mode and pair stars "
        "(see the <a href=\"topic:astrometry\">astrometry</a> topic).</li>"
        "<li>Press " + _key(c + " + Z") + " to fit; repeat until the solution stabilises. Then "
        "<a href=\"topic:residuals\">check the residual plots</a> and save (" + _key(c + " + S") + ").</li>"
        "</ol>"
        + _nav_links(next_pair=('astrometry', 'Calibrate astrometry'),
                     related=[('residuals', 'Checking the fit'), ('tabs', 'Guide to the tabs')])
    )
    return _page("Overview &amp; quick start", body)


def _topic_levels(gui):
    c = _ctrl(gui)
    body = (
        "<p class=\"lead\">The <b>Levels</b> tab controls the <b>display contrast</b> of the image. "
        "It changes only how the image looks on screen &ndash; never the underlying pixel data, the "
        "fit, or the photometry.</p>"

        "<h3>The histogram</h3>"
        "<p>The plot is a <b>histogram</b>: for each brightness value (horizontal axis) it shows how "
        "many pixels in the image have that value (height). Most pixels are dark sky, so there is "
        "usually a tall spike near the low end, with a long thin tail of brighter pixels (stars).</p>"

        "<h3>Black point &amp; white point</h3>"
        "<p>Two draggable handles bound a highlighted region on the histogram:</p>"
        "<ul>"
        "<li><b>Black point</b> (lower handle): every pixel at or below this value is shown as pure "
        "black.</li>"
        "<li><b>White point</b> (upper handle): every pixel at or above this value is shown as pure "
        "white.</li>"
        "</ul>"
        "<p>Brightness between the two is spread across the grey ramp. <b>Narrowing</b> the window "
        "(handles closer together) increases contrast and makes faint stars pop; <b>widening</b> it "
        "softens the image. Drag the edges of the highlighted region to set them by hand.</p>"

        "<h3>Auto levels (" + _key(c + " + A") + ")</h3>"
        "<p>Auto levels sets the black and white points automatically from the image's brightness "
        "distribution &ndash; roughly the <b>0.1st percentile</b> for black and the "
        "<b>99.95th percentile</b> for white, while ignoring the brightest few percent of pixels so "
        "hot or saturated pixels don't blow out the stretch. It is a good starting point for almost "
        "any image. Press " + _key(c + " + A") + " again to return to your manual levels; while auto "
        "is on, the handles are locked.</p>"
        + _callout("Levels are display-only. Set them so you can comfortably see the stars you need "
                   "to pick &ndash; they have no effect on the calibration result.")
        + _nav_links(related=[('tabs', 'Guide to the tabs')])
    )
    return _page("Levels (display contrast)", body)


def _topic_tabs(gui):
    mode = _mode(gui)
    mode_name = "SkyFit" if mode == 'skyfit' else "Manual Reduction"

    rows = []
    rows.append(("Levels",
                 "Adjust image brightness and contrast with the histogram &ndash; drag the handles, "
                 "or press " + _key(_ctrl(gui) + " + A") + " for auto levels. "
                 "<a href=\"topic:levels\">More</a>."))

    if mode == 'skyfit':
        rows.append(("Fit Parameters",
                     "The main calibration tab. Holds the pointing and distortion values and the "
                     "<b>Fit</b>, <b>Auto Fit</b> and <b>Auto Pointing</b> buttons, the distortion "
                     "model, the refraction / equal-aspect / asymmetry options, and the photometry "
                     "fit."))
        rows.append(("Station",
                     "Observer location (latitude, longitude, elevation), buttons to nudge the "
                     "station position, and - when geo points are loaded - reload and auto-refit "
                     "controls."))

    rows.append(("Star Detection",
                 "Re-detect stars on the current image with tunable parameters and use them instead "
                 "of the stored CALSTARS detections."))
    rows.append(("Mask",
                 "Draw or paint a mask to hide obstructions (roofs, trees) from the star catalog."))
    rows.append(("Settings",
                 "Display options: which overlays to show (catalog / detected / selected stars, "
                 "constellations, coordinate grids, distortion), image gamma, magnitude limits, "
                 "invert colours and more."))

    if mode == 'manualreduction' and _is_dfn(gui):
        rows.append(("Debruijn",
                     "Recover the time of a DFN fireball from its shutter-break sequence."))

    rows.append(("Help", "This guide."))

    body = (
        "<p class=\"lead\">The tabs run down the right-hand edge of the window. Click a tab to open "
        "it, and click it again to collapse the panel. Which tabs appear depends on the mode "
        "(currently <b>" + mode_name + "</b>).</p>"
        + _defn_table(rows)
    )
    return _page("Guide to the tabs", body)


def _topic_astrometry(gui):
    c = _ctrl(gui)
    body = (
        "<p>The goal is to pair at least <b>14 catalog stars</b> spread uniformly across the whole "
        "image (more is better, and include some near the horizon). Then fit.</p>"
        "<h3>Picking stars</h3>"
        "<ol>"
        "<li>Press " + _key(c + " + R") + " to start. Two yellow circles follow the cursor; scroll "
        "to resize them.</li>"
        "<li>Click a star so it sits <b>fully inside the inner circle</b> - a yellow + marks the "
        "centroid. Use " + _key(c + " + Left click") + " to force a manual position instead.</li>"
        "<li>A purple + marks the likely catalog match. If it is wrong, click the correct catalog "
        "star to move it.</li>"
        "<li>Press " + _key("ENTER") + " or " + _key("SPACE") + " to accept the pair (marker turns "
        "into a blue X). Right-click a pair to remove it.</li>"
        "</ol>"
        "<h3>Fitting</h3>"
        "<p>Press " + _key(c + " + Z") + " to fit, several times until it stops changing. Minimum "
        "stars: <b>5</b> for pointing, <b>~12</b> to also fit distortion.</p>"
        "<h3>Fit options (Fit Parameters tab)</h3>"
        "<ul>"
        "<li><b>Refraction</b>: keep on for all cameras.</li>"
        "<li><b>Equal aspect</b>: turn off for non-square pixels (radial models only).</li>"
        "<li><b>Asymmetry correction</b>: enable if the lens is not flat against the sensor.</li>"
        "<li><b>Distortion model</b> (" + _key(c + " + 1..6") + "): start with the lowest radial "
        "order and increase only if residual structure remains.</li>"
        "<li><b>Only fit pointing</b>: re-fits pointing from your picked stars without touching "
        "distortion - use it when re-calibrating a fireball clip on a known lens. With this checked, "
        "the <b>Fit</b> button does a pure pointing fit (no distortion, no astrometry.net).</li>"
        "</ul>"
        "<p><b>Note on the auto buttons:</b> <b>Auto Fit</b> and <b>Auto Pointing</b> work from "
        "automatically detected stars (and astrometry.net), <i>not</i> from your manual picks, and "
        "they do not honour \"Only fit pointing\". For a picked-star pointing-only fit, use the "
        "<b>Fit</b> button.</p>"
        "<p>Press " + _key("L") + " for the astrometry residual plot.</p>"
        + _shortcut_table([
            ("LEFT CLICK", "Centroid the star under the cursor"),
            (c + " + LEFT CLICK", "Manual (forced) star position"),
            ("ENTER / SPACE", "Accept the star pair"),
            (c + " + SPACE", "Mark the pair as bad"),
            ("SHIFT + SPACE", "Jump to a random region"),
            ("RIGHT CLICK", "Remove a pair"),
            (c + " + SCROLL", "Adjust aperture radius"),
            (c + " + Z", "Fit the plate"),
            (c + " + SHIFT + Z", "Fit with distortion params reset to 0"),
        ])
        + _nav_links(next_pair=('photometry', 'Photometry'),
                     related=[('residuals', 'Checking the fit')])
    )
    return _page("Astrometric calibration", body)


def _topic_photometry(gui):
    c = _ctrl(gui)
    body = (
        "<p>Photometry converts pixel intensity into stellar magnitude, which is needed to estimate "
        "meteor brightness and mass. It is fitted <b>automatically</b> from your calibration stars; "
        "press " + _key("P") + " (or the <b>Photometry</b> button) to review and tune it. See "
        "<a href=\"topic:residuals\">checking the fit</a> for how to read the plot.</p>"
        "<ol>"
        "<li>Press " + _key("P") + " (or Fit Parameters &gt; Photometry) to show the magnitude fit.</li>"
        "<li>Adjust <b>extinction</b> (typically 0.6-1.0) until the points follow the curved line.</li>"
        "<li>Set the camera <b>gamma</b> correctly (science cameras 1.0, consumer ~0.45) - it is "
        "essential for good photometry. Star photometry through SkyFit can reveal the true gamma.</li>"
        "<li>Only enable <b>fixed vignetting</b> for cameras with well-measured vignetting.</li>"
        "<li>Remove saturated / high-error stars and re-fit until the scatter is small "
        "(aim for &lt;= ~0.2 mag).</li>"
        "</ol>"
        "<p>For saturated-object work you may want to launch with <b>--nobg</b> (no background "
        "subtraction) or <b>--peribg</b> (peripheral background).</p>"
        + _shortcut_table([
            ("P", "Show / refresh the photometry fit"),
            ("U / J", "Image display gamma"),
            (c + " + D", "Load a dark frame"),
            (c + " + F", "Load a flat field"),
        ])
        + _nav_links(next_pair=('residuals', 'Checking the fit'),
                     related=[('astrometry', 'Calibrate astrometry')])
    )
    return _page("Photometry", body)


def _topic_residuals(gui):
    body = (
        "<p class=\"lead\">After a fit, always check the two residual plots. Open them with the "
        "<b>Astrometry</b> and <b>Photometry</b> buttons in the Fit Parameters tab (or press "
        + _key("L") + " and " + _key("P") + "). The small <b>i</b> button next to them reopens this "
        "page.</p>"

        "<h3>Astrometry residuals &ndash; what good looks like</h3>"
        "<ul>"
        "<li><b>Small residuals.</b> Aim for an RMSD <b>below about 0.2 px</b>. A star slightly "
        "higher is fine; a few large outliers usually mean a mis-paired star &ndash; remove them and "
        "re-fit.</li>"
        "<li><b>Good coverage.</b> Use stars across the <i>whole</i> field of view &ndash; corners "
        "and edges, not just the centre &ndash; and over a range of radii from the image centre. "
        "Gaps in coverage let the distortion drift where there are no stars.</li>"
        "<li><b>No bias or trend.</b> The residual vectors should point in random directions, not "
        "all the same way (a pointing bias) and not swirling around the centre (a rotation or scale "
        "error).</li>"
        "<li><b>Flat error vs radius.</b> This is the important one: the plot of error against "
        "radius from centre should be <b>flat &ndash; no slope, curve or wiggle</b>. A trend means "
        "the distortion model is wrong: raise the radial order if the error grows toward the edge, "
        "or lower it if a high-order model is over-fitting (wiggles).</li>"
        "</ul>"

        "<h3>Photometry residuals &ndash; what good looks like</h3>"
        "<ul>"
        "<li><b>Flat residuals vs radius (corrected).</b> After the vignetting correction, the "
        "magnitude residuals plotted against radius should sit <b>flat around the zero point</b> "
        "&ndash; no slope or curve with radius.</li>"
        "<li><b>Vignetting follows the drop-off.</b> The vignetting curve should track the measured "
        "fall-off in star brightness toward the edges of the image.</li>"
        "<li><b>Correct gamma.</b> The magnitude fit (measured vs catalog magnitude) should be "
        "<b>flat</b>. If the camera <b>gamma</b> is wrong the plot develops a <b>slope</b> instead "
        "of being flat &ndash; adjust gamma until it flattens. The right gamma is essential for good "
        "photometry.</li>"
        "</ul>"

        "<h3>Removing a bad star</h3>"
        "<p>Spotted an outlier in a plot? Remove it without leaving the plot:</p>"
        "<ol>"
        "<li>Click the offending point in the <b>Astrometry</b> or <b>Photometry</b> plot &ndash; a "
        "marker appears on that star in the image.</li>"
        "<li>Right-click that marker in the image to remove the pair.</li>"
        "</ol>"
        "<p>The photometry fit and its plot update automatically. The astrometry residuals update "
        "after you re-fit the plate (" + _key(_ctrl(gui) + " + Z") + ").</p>"
        + _nav_links(related=[('astrometry', 'Calibrate astrometry'), ('photometry', 'Photometry')])
    )
    return _page("Checking the fit (residual plots)", body)


def _topic_calibration_files(gui):
    c = _ctrl(gui)
    body = (
        "<p>Calibration frames improve both astrometry and photometry. Load them from the File "
        "Manager / Calibration Files dialog or with shortcuts.</p>"
        + _shortcut_table([
            (c + " + D", "Load a dark frame"),
            (c + " + F", "Load a flat field"),
            (c + " + A", "Auto-adjust display levels"),
        ])
        + "<p>A <b>mask</b> hides parts of the image (e.g. obstructions) from the star catalog - "
        "see the <a href=\"topic:mask\">mask drawing</a> topic. Flat/dark are applied to the image "
        "before measurement; the <b>--flatbiassub</b> option also subtracts bias from the flat.</p>"
    )
    return _page("Calibration files (dark / flat / mask)", body)


def _topic_stardetect(gui):
    params = _defn_table([
        ("Intensity threshold <span class=\"tip\">(def. 18)</span>",
         "How far above the local background a pixel must rise to count as a star. <b>Lower</b> "
         "detects fainter stars but also more noise; <b>higher</b> keeps only the bright, confident "
         "ones. The single most useful knob &ndash; tune it first."),
        ("Neighborhood size <span class=\"tip\">(def. 10 px)</span>",
         "Size of the local window used to pick one peak per star. <b>Larger</b> merges close stars "
         "(fewer detections); <b>smaller</b> separates them but can split one bright star into "
         "several. Set it a little larger than your typical star spacing."),
        ("Max stars <span class=\"tip\">(def. 200)</span>",
         "Upper limit on how many detections are kept (brightest first). Raise it for rich, "
         "wide-field images; lower it to keep only the brightest."),
        ("Gamma <span class=\"tip\">(def. 1.0)</span>",
         "Gamma stretch applied to the image <i>for detection only</i> (not the camera gamma and "
         "not the display gamma). Values below 1 lift faint stars out of the background so they get "
         "detected."),
        ("Segment radius <span class=\"tip\">(def. 4 px)</span>",
         "Radius of the patch used to centroid and measure each star. Match it to the typical star "
         "size (FWHM): too small clips the star and worsens the centroid; too large pulls in "
         "neighbours and background."),
        ("Max feature ratio <span class=\"tip\">(def. 0.8)</span>",
         "Rejects extended / elongated blobs (clouds, galaxies, hot rows, aircraft). <b>Lower</b> is "
         "stricter and throws away more non-star features; raise it if real stars are being "
         "rejected."),
        ("Roundness threshold <span class=\"tip\">(def. 0.5)</span>",
         "Rejects detections that are not round enough (streaks, edges, partially merged stars). "
         "<b>Higher</b> demands rounder shapes; lower is more permissive."),
        ("Catalog LM",
         "Limiting magnitude of the catalog used when comparing detected vs catalog star counts "
         "(used by Tune)."),
    ])

    body = (
        "<p class=\"lead\">By default SkyFit uses the star detections stored in the CALSTARS file. "
        "This tab lets you <b>re-detect</b> stars on the current image with your own parameters and "
        "use those instead &ndash; handy when the default detection misses stars or picks up "
        "noise.</p>"

        "<h3>How to use it</h3>"
        "<ol>"
        "<li>Adjust the parameters below and click <b>Redetect</b> to re-run detection on the "
        "current image (or <b>Redetect all</b> for every image). Detected stars are drawn on the "
        "image so you can see the effect.</li>"
        "<li>Click <b>Tune</b> to let SkyFit balance the parameters and catalog limiting magnitude "
        "so the number of detected stars roughly matches the catalog stars in the field.</li>"
        "<li>Tick <b>Use Override Detections</b> to feed these detections into fitting and Auto "
        "Fit instead of CALSTARS.</li>"
        "<li><b>Save to Config</b> writes the parameters to your config file so future runs reuse "
        "them.</li>"
        "</ol>"

        "<h3>Parameters</h3>"
        + params +
        "<p>Goal: detect the real stars across the whole frame without picking up noise or extended "
        "features. If you see noise specks, raise the intensity threshold or tighten the roundness / "
        "feature-ratio limits; if faint stars are missed, lower the threshold or the detection "
        "gamma.</p>"
    )
    return _page("Star detection override", body)


def _topic_mask(gui):
    body = (
        "<p>The <b>Mask</b> tab edits the mask applied to the star catalog so stars over "
        "obstructions (roofs, trees) are ignored.</p>"
        "<ul>"
        "<li><b>Draw mode</b>: click to lay down polygon vertices; the enclosed area is masked.</li>"
        "<li><b>Brush mode</b>: paint/erase the mask freehand; adjust the brush size.</li>"
        "<li>Undo strokes, clear polygons, invert, and toggle the overlay from the same tab.</li>"
        "</ul>"
        "<p>Launch with <b>-m / --mask PATH</b> to start from an existing mask.</p>"
    )
    return _page("Mask drawing", body)


def _topic_settings(gui):
    mode = _mode(gui)

    display_rows = [
        ("avepixel / maxpixel", "Show the average frame or the maximum-value frame."),
        ("Show Catalog Stars", "Overlay catalog star positions."),
        ("Show Spectral Type", "Colour catalog stars by spectral type."),
        ("Show Star Names", "Label the brighter catalog stars."),
        ("Label Mag Limit", "Faintest magnitude that still gets a star label."),
        ("Show Constellation Lines", "Draw the constellation stick figures."),
        ("Show Detected Stars", "Overlay the automatically detected stars (CALSTARS)."),
        ("Show Selected Stars", "Overlay the stars you have paired for the fit."),
        ("Show Distortion", "Draw the lens-distortion grid."),
        ("Invert Colors", "Invert the image (dark stars on a light background)."),
        ("Auto Pan To Next Star", "Recentre the view on the next star while picking."),
        ("Single Click Photometry", "Measure photometry with a single click while picking."),
    ]

    catalog_rows = [
        ("Gamma", "Display gamma of the image (visual only &ndash; not the camera gamma)."),
        ("Lim Mag", "Catalog limiting magnitude: how faint to load catalog stars."),
        ("Correct Mag for Ext./Vign.", "Apply extinction and vignetting correction to catalog "
         "magnitudes before filtering."),
        ("Filter Mag Err", "Drop paired stars whose magnitude error exceeds this value when "
         "fitting."),
        ("Star Catalog", "Which star catalog to use (e.g. GAIA, BSC5)."),
        ("Grid", "Overlay no grid, an RA/Dec grid, or an Az/Alt grid."),
    ]

    sat_rows = [
        ("Show Satellite Tracks", "Overlay predicted satellite passes. Needs TLEs (internet to "
         "download, unless a TLE file is loaded)."),
        ("Automatically compute tracks", "Recompute tracks when you change frame. Off by default "
         "for speed."),
        ("Load TLE File / Reset TLE Selection", "Use a local TLE file instead of downloading, or "
         "clear that choice."),
        ("Redraw Satellite Tracks", "Recompute and redraw the tracks now."),
    ]

    body = (
        "<p class=\"lead\">The Settings tab controls what is drawn on the image, plus a few catalog "
        "and satellite options. Toggles take effect immediately.</p>"
        "<h3>Display overlays</h3>" + _defn_table(display_rows)
        + "<h3>Image, catalog &amp; magnitudes</h3>" + _defn_table(catalog_rows)
        + "<h3>Satellites</h3>"
        "<p>Turn these on here when you need them &ndash; see <a href=\"topic:sattracks\">satellite "
        "tracks</a> for the full details.</p>" + _defn_table(sat_rows)
    )

    if mode == 'manualreduction':
        mr_rows = [
            ("Show Picks", "Show your frame-by-frame meteor picks."),
            ("Show Great Circle Line", "Draw the great-circle path fitted through the picks."),
            ("Show Photometry Highlight", "Highlight the photometry aperture region."),
        ]
        body += "<h3>Manual reduction display</h3>" + _defn_table(mr_rows)

    if _has_geopoints(gui):
        body += ("<h3>Geo points</h3>" + _defn_table([
            ("Measure ground points", "Pick ground positions instead of sky positions "
             "(see <a href=\"topic:geopoints\">geo points</a>)."),
        ]))

    return _page("Settings tab", body)


def _topic_station(gui):
    body = (
        "<p class=\"lead\">The Station tab holds the observer's location and, optionally, "
        "terrestrial geo points.</p>"
        "<h3>Station coordinates</h3>"
        "<p>Enter the camera's <b>latitude</b>, <b>longitude</b> and <b>elevation</b> as precisely "
        "as you can. These set the observer position used for every sky calculation; for meteor "
        "work an accurate location is essential for triangulation between stations.</p>"
        "<h3>Move station</h3>"
        "<p>The movement buttons nudge the station position by small steps &ndash; useful for "
        "fine-tuning geo-point alignment. <b>Auto refit astrometry</b> re-fits the plate "
        "automatically whenever you move the station.</p>"
    )
    if _has_geopoints(gui):
        body += _callout("Geo points are loaded. See <a href=\"topic:geopoints\">Geo points</a> for "
                         "how to calibrate pointing from terrestrial landmarks.")
    body += _nav_links(related=[('geopoints', 'Geo points'), ('astrometry', 'Calibrate astrometry')])
    return _page("Station &amp; location", body)


def _topic_geopoints(gui):
    body = (
        "<p>Geo points are <b>fixed terrestrial landmarks</b> (from a name,lat,lon,elevation file, "
        "passed with <b>-p / --geopoints</b>) projected onto the image from the observer's position. "
        "They let you calibrate pointing from distant ground references instead of - or in addition "
        "to - stars, which is handy for daytime or star-poor scenes.</p>"
        "<h3>Workflow</h3>"
        "<ol>"
        "<li>Enter the station coordinates precisely in the <b>Station</b> tab.</li>"
        "<li>Enter star-picking mode (" + _key(_ctrl(gui) + " + R") + ") and click near a projected "
        "geo point to pair it, the same way you pair stars.</li>"
        "<li>Check <b>Only fit pointing</b> and press " + _btn("Fit") + " to refine pointing from "
        "the picked geo points without disturbing the distortion.</li>"
        "</ol>"
        + _callout("The " + _btn("Auto Fit") + " and " + _btn("Auto Pointing") + " buttons work "
        "from automatically <i>detected stars</i> and will replace your picks &ndash; they cannot "
        "use geo points. For a geo-point solution always use the " + _btn("Fit") + " button.", "note")
        + "<p>The Station tab also has movement buttons (to nudge the station position) and an "
        "<b>Auto refit astrometry</b> option that re-fits when the station is moved. "
        "Toggle <b>Measure ground points</b> to pick ground vs sky positions.</p>"
        + _nav_links(related=[('station', 'Station & location'), ('astrometry', 'Calibrate astrometry')])
    )
    return _page("Geo points (ground references)", body)


def _topic_sattracks(gui):
    c = _ctrl(gui)
    body = (
        "<p>Satellite tracks overlay predicted satellite paths on the image (enabled with "
        "<b>--sattracks</b>, optionally with <b>--tle_file</b>). Downloading TLEs needs internet; a "
        "local TLE file closest to the clip start is selected automatically when provided.</p>"
        + _shortcut_table([
            (c + " + T", "Toggle satellite tracks"),
        ])
    )
    return _page("Satellite tracks", body)


def _topic_frfiles(gui):
    body = (
        "<p>FR (fast-read) files hold the bright pixels of detected events. Launching with "
        "<b>-r / --fr</b> loads them alongside the FF files so you can inspect and reduce the "
        "captured meteor frames. FR-line navigation is available in Manual Reduction mode "
        "(see the <a href=\"topic:shortcuts_mr\">Manual Reduction shortcuts</a>).</p>"
    )
    return _page("FR files", body)


def _topic_shortcuts_skyfit(gui):
    c = _ctrl(gui)
    nav = _shortcut_table([
        ("F1", "Show / hide the on-image info panel"),
        ("Left / Right", "Previous / next image"),
        (c + " + Left / Right", "+/- 10 images"),
        ("Scroll", "Zoom in / out"),
        ("V", "Centre on field of view"),
    ])
    pointing = _shortcut_table([
        ("A / D", "Azimuth"),
        ("S / W", "Altitude"),
        ("Q / E", "Position angle"),
        ("Up / Down", "Scale"),
        ("1/2, 3/4", "X / Y offset"),
        ("5/6, 7/8", "X / Y 1st distortion coeff."),
        ("9 / 0", "Extinction scale"),
        ("T", "Toggle refraction correction"),
        ("G / Y / B", "Equal aspect / asymmetry / dist=centre (radial only)"),
        (c + " + 1..6", "Distortion model: poly3+radial, poly3+radial3, radial3/5/7/9"),
    ])
    display = _shortcut_table([
        ("R / F", "Limiting magnitude"),
        ("+ / -", "Increment step"),
        ("M", "Toggle maxpixel / avepixel"),
        ("H", "Show / hide catalog stars"),
        ("C", "Show / hide detected stars"),
        (c + " + I", "Show / hide distortion"),
        ("U / J", "Image gamma"),
        ("I", "Invert colours"),
        (c + " + A", "Auto levels"),
        (c + " + G", "Cycle coordinate grids"),
        (c + " + T", "Toggle satellite tracks"),
        ("SHIFT + Z", "Show zoomed window"),
    ])
    actions = _shortcut_table([
        (c + " + R", "Enter / exit star picking"),
        (c + " + Z", "Fit plate"),
        (c + " + SHIFT + Z", "Fit with distortion reset to 0"),
        ("L", "Astrometry residual plot"),
        ("P", "Photometry fit plot"),
        (c + " + D / " + c + " + F", "Load dark / flat"),
        (c + " + U / " + c + " + O", "Pan to next star / toggle auto-pan"),
        (c + " + X", "astrometry.net (image upload)"),
        (c + " + SHIFT + X", "astrometry.net (star XY only)"),
        (c + " + SHIFT + B", "Fit spectral bands"),
        (c + " + N", "New platepar"),
        (c + " + S", "Save platepar &amp; state"),
        (c + " + SHIFT + S", "Save platepar to data folder"),
    ])
    body = ("<h3>Navigation</h3>" + nav + "<h3>Pointing &amp; distortion</h3>" + pointing
            + "<h3>Display</h3>" + display + "<h3>Fitting &amp; files</h3>" + actions)
    return _page("Keyboard reference - SkyFit", body)


# ----- Manual Reduction topics ---------------------------------------------------------------- #

def _topic_mr_overview(gui):
    c = _ctrl(gui)
    body = (
        "<p>Manual Reduction is for measuring a meteor or fireball frame by frame: you mark its "
        "position on each frame, build a light curve, and export the measurements.</p>"
        "<h3>Quick start</h3>"
        "<ol>"
        "<li>Load a platepar with " + _key(c + " + P") + " so positions get sky coordinates.</li>"
        "<li>Press " + _key(c + " + R") + " to start picking (see "
        "<a href=\"topic:mr_picking\">picking</a>).</li>"
        "<li>Step through frames with " + _key("Left / Right") + " and mark the meteor.</li>"
        "<li>Press " + _key("P") + " for the <a href=\"topic:mr_lightcurve\">light curve</a>.</li>"
        "<li>Save with " + _key(c + " + S") + " (FTPdetectinfo).</li>"
        "</ol>"
        + _nav_links(next_pair=('mr_picking', 'Pick meteor positions'),
                     related=[('mr_lightcurve', 'Light curve & saving')])
    )
    return _page("Overview &amp; quick start", body)


def _topic_mr_picking(gui):
    c = _ctrl(gui)
    body = (
        "<p>Press " + _key(c + " + R") + " to enter picking mode, then mark the meteor on each "
        "frame, keeping the points in time order.</p>"
        + _shortcut_table([
            ("LEFT CLICK", "Centroid the meteor at the cursor"),
            (c + " + LEFT CLICK", "Force a pick at the exact cursor position"),
            ("ALT / Num0 + LEFT CLICK", "Mark a gap (DFN sequences)"),
            ("Left / Right", "Previous / next frame"),
            (c + " + Left / Right", "+/- 10 frames"),
            ("Down / Up", "+/- 25 frames"),
            (", / .", "Previous / next FR line"),
            ("M", "Show maxpixel"),
            ("K", "Subtract average"),
            ("T", "Toggle refraction correction"),
        ])
        + _nav_links(next_pair=('mr_lightcurve', 'Light curve & saving'))
    )
    return _page("Picking meteor positions", body)


def _topic_mr_lightcurve(gui):
    c = _ctrl(gui)
    body = (
        "<p>Press " + _key("P") + " to show the light curve (intensity vs frame) of your picks.</p>"
        "<p>Save your reduction with " + _key(c + " + S") + " - this writes an FTPdetectinfo file "
        "(named <i>..._manual.txt</i>) in the data folder, applying the platepar astrometry to each "
        "centroid. Save the current frame image with " + _key(c + " + W") + ".</p>"
    )
    return _page("Light curve &amp; saving", body)


def _topic_mr_astra(gui):
    c = _ctrl(gui)
    body = (
        "<p><b>ASTRA</b> (Astrometric Streak Tracking and Refinement Algorithm) automates EMCCD "
        "picking/photometry and can also refine manual picks. Open it with " + _key(c + " + K") + ".</p>"
        "<p>ASTRA needs at least 3 frame-adjacent leading-edge picks at a good-SNR section, plus 2 "
        "leading-edge picks at the start/end frames of the event. These can be loaded from ECSV/txt "
        "or made manually. Hover over its parameters and READY/NOT READY icons for guidance.</p>"
    )
    return _page("ASTRA (automated picking)", body)


def _topic_debruijn(gui):
    body = (
        "<p>For DFN fireball videos the timing is encoded as a de Bruijn sequence of shutter breaks. "
        "Mark 10-20 points (use ALT/Num0 + click for gaps), then open the <b>Debruijn</b> tab and "
        "press <b>Check Sequence</b> - it searches for a unique time solution. Set the time-direction "
        "option if known.</p>"
    )
    return _page("DFN / Debruijn timing", body)


def _topic_shortcuts_mr(gui):
    c = _ctrl(gui)
    nav = _shortcut_table([
        ("F1", "Show / hide the on-image info panel"),
        ("Left / Right", "Previous / next frame"),
        (c + " + Left / Right", "+/- 10 frames"),
        ("Down / Up", "+/- 25 frames"),
        (", / .", "Previous / next FR line"),
        ("Scroll", "Zoom in / out"),
    ])
    actions = _shortcut_table([
        (c + " + R", "Enter / exit picking"),
        ("LEFT CLICK", "Centroid"),
        (c + " + LEFT CLICK", "Force pick"),
        ("ALT / Num0 + LEFT CLICK", "Mark gap (DFN)"),
        ("M", "Show maxpixel"),
        ("K", "Subtract average"),
        ("T", "Toggle refraction correction"),
        ("U / J", "Image gamma"),
        ("P", "Show light curve"),
        (c + " + A", "Auto levels"),
        (c + " + D / " + c + " + F", "Load dark / flat"),
        (c + " + P", "Load platepar"),
        (c + " + W", "Save current frame"),
        (c + " + S", "Save FTPdetectinfo"),
        (c + " + K", "Open ASTRA GUI"),
    ])
    body = "<h3>Navigation</h3>" + nav + "<h3>Picking &amp; files</h3>" + actions
    return _page("Keyboard reference - Manual Reduction", body)


# ---------------------------------------------------------------------------------------------- #

# Ordered topic registry. Keep the order you want them listed on the Home page.
HELP_TOPICS = [
    # SkyFit
    ('overview',          dict(title="Overview &amp; quick start",        modes=('skyfit',),          enabled=_always,        build=_topic_overview,
                               desc="Start here: what SkyFit does and the fastest way to calibrate.")),
    ('tabs',              dict(title="Guide to the tabs",                 modes=('skyfit',),          enabled=_always,        build=_topic_tabs,
                               desc="What each tab on the right does.")),
    ('levels',            dict(title="Levels (display contrast)",         modes=('skyfit',),          enabled=_always,        build=_topic_levels,
                               desc="The histogram, black/white points, auto levels.")),
    ('astrometry',        dict(title="Calibrate astrometry (pick stars)", modes=('skyfit',),          enabled=_always,        build=_topic_astrometry,
                               desc="Pick stars and fit the plate by hand.")),
    ('photometry',        dict(title="Photometry",                        modes=('skyfit',),          enabled=_always,        build=_topic_photometry,
                               desc="Calibrate brightness: extinction, vignetting, gamma.")),
    ('residuals',         dict(title="Checking the fit (residual plots)",  modes=('skyfit',),          enabled=_always,        build=_topic_residuals,
                               desc="Read the residual plots and the values to hit.")),
    ('calibration_files', dict(title="Calibration files (dark/flat/mask)",modes=('skyfit',),          enabled=_always,        build=_topic_calibration_files,
                               desc="Load dark, flat and mask frames.")),
    ('station',           dict(title="Station &amp; location",            modes=('skyfit',),          enabled=_always,        build=_topic_station,
                               desc="Observer location, station moves, auto-refit.")),
    ('geopoints',         dict(title="Geo points (ground references)",    modes=('skyfit',),          enabled=_has_geopoints, build=_topic_geopoints,
                               desc="Calibrate pointing from terrestrial landmarks.")),
    ('sattracks',         dict(title="Satellite tracks",                  modes=('skyfit',),          enabled=_always,        build=_topic_sattracks,
                               desc="Overlay predicted satellite passes (toggle in Settings).")),
    ('frfiles',           dict(title="FR files",                          modes=('skyfit',),          enabled=_has_fr,        build=_topic_frfiles,
                               desc="Work with fast-read meteor detection files.")),
    ('stardetect',        dict(title="Star detection override",           modes=('skyfit',),          enabled=_always,        build=_topic_stardetect,
                               desc="Re-detect stars with tunable parameters.")),
    ('mask',              dict(title="Mask drawing",                      modes=('skyfit',),          enabled=_always,        build=_topic_mask,
                               desc="Hide obstructions from the star catalog.")),
    ('settings',          dict(title="Settings tab",                      modes=('skyfit',),          enabled=_always,        build=_topic_settings,
                               desc="Every option in the Settings tab explained.")),
    ('shortcuts_skyfit',  dict(title="Keyboard reference",                modes=('skyfit',),          enabled=_always,        build=_topic_shortcuts_skyfit,
                               desc="Every keyboard shortcut, grouped.")),

    # Manual Reduction
    ('mr_overview',       dict(title="Overview &amp; quick start",        modes=('manualreduction',), enabled=_always,        build=_topic_mr_overview,
                               desc="Start here: measure a meteor frame by frame.")),
    ('mr_tabs',           dict(title="Guide to the tabs",                 modes=('manualreduction',), enabled=_always,        build=_topic_tabs,
                               desc="What each tab on the right does.")),
    ('mr_levels',         dict(title="Levels (display contrast)",         modes=('manualreduction',), enabled=_always,        build=_topic_levels,
                               desc="The histogram, black/white points, auto levels.")),
    ('mr_picking',        dict(title="Pick meteor positions",             modes=('manualreduction',), enabled=_always,        build=_topic_mr_picking,
                               desc="Mark the meteor position on each frame.")),
    ('mr_lightcurve',     dict(title="Light curve &amp; saving",          modes=('manualreduction',), enabled=_always,        build=_topic_mr_lightcurve,
                               desc="View the light curve and export results.")),
    ('debruijn',          dict(title="DFN / Debruijn timing",             modes=('manualreduction',), enabled=_is_dfn,        build=_topic_debruijn,
                               desc="Recover DFN fireball timing.")),
    ('mr_astra',          dict(title="ASTRA (automated picking)",         modes=('manualreduction',), enabled=_always,        build=_topic_mr_astra,
                               desc="Automate or refine picks with ASTRA.")),
    ('mr_settings',       dict(title="Settings tab",                      modes=('manualreduction',), enabled=_always,        build=_topic_settings,
                               desc="Every option in the Settings tab explained.")),
    ('shortcuts_mr',      dict(title="Keyboard reference",                modes=('manualreduction',), enabled=_always,        build=_topic_shortcuts_mr,
                               desc="Every keyboard shortcut, grouped.")),
]

_TOPIC_MAP = dict(HELP_TOPICS)


def _visible_topics(gui):
    """ Return the (id, meta) topics for the current mode whose feature gate is satisfied. """
    mode = _mode(gui)
    visible = []
    for topic_id, meta in HELP_TOPICS:
        if mode not in meta['modes']:
            continue
        try:
            if not meta['enabled'](gui):
                continue
        except Exception:
            continue
        visible.append((topic_id, meta))
    return visible


def buildHelpHome(gui, query=None):
    """ Build the Help home page: short intro + triage links for the current mode/features.

    If ``query`` is given, the topic list is filtered to titles/descriptions containing it.
    """

    mode = _mode(gui)
    mode_name = "SkyFit" if mode == 'skyfit' else "Manual Reduction"

    if mode == 'skyfit':
        intro = ("<p class=\"lead\">SkyFit turns an image into a calibrated <b>platepar</b> (plate "
                 "parameters &ndash; sky coordinates and magnitudes). New here? Start with "
                 "<a href=\"topic:overview\">Overview &amp; quick start</a>.</p>")
    else:
        intro = ("<p class=\"lead\">Manual Reduction measures a meteor frame by frame and exports "
                 "the result. New here? Start with <a href=\"topic:mr_overview\">Overview &amp; "
                 "quick start</a>.</p>")

    topics = _visible_topics(gui)
    if query:
        q = query.lower()
        topics = [(tid, m) for (tid, m) in topics
                  if q in m['title'].lower() or q in m.get('desc', '').lower()]

    rows = ""
    for topic_id, meta in topics:
        rows += ('<tr><td valign="top"><a href="topic:{tid}">{title}</a></td>'
                 '<td valign="top" class="desc">{desc}</td></tr>').format(
                     tid=topic_id, title=meta['title'], desc=meta.get('desc', ''))

    if query:
        section = "<h3>Search results</h3>"
        if not rows:
            section += "<p class=\"lead\">No topics match &ldquo;{:s}&rdquo;.</p>".format(query)
        body = section + "<table cellspacing=\"0\" cellpadding=\"0\">" + rows + "</table>"
    else:
        body = (
            intro
            + "<h3>What do you want to do?</h3>"
            "<table cellspacing=\"0\" cellpadding=\"0\">" + rows + "</table>"
            "<hr>"
            "<p class=\"lead\">Every tab has an <b>i</b> button in its top-right corner for help on "
            "that tab. Open this guide any time from the <b>Help</b> menu or <b>Shift+F1</b>; "
            "<b>F1</b> shows/hides the on-image info panel.</p>"
            "<p>Switch between SkyFit and Manual Reduction with the buttons under the image; this "
            "Help updates to match.</p>"
            "<p>Full online manual: <a href=\"" + WIKI_URL + "\">GMN SkyFit2 wiki</a>.</p>"
        )

    return "<h2>SkyFit2 Help &mdash; {mode}</h2>{body}".format(mode=mode_name, body=body)


def buildHelpTopic(gui, topic_id):
    """ Build the detail HTML for one topic id, or None if unknown. """

    meta = _TOPIC_MAP.get(topic_id)
    if meta is None:
        return None
    try:
        return meta['build'](gui)
    except Exception as e:
        return _page(meta.get('title', 'Help'), "<p>Could not render this help topic: {:s}</p>".format(str(e)))
