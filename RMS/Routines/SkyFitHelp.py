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


def _two_col_table(rows, term_fn):
    """ Build a two-column table; term_fn formats the left cell of each (term, description) row. """
    html = '<table cellspacing="0" cellpadding="0">'
    for term, desc in rows:
        html += ('<tr><td valign="top">{t}</td>'
                 '<td valign="top" class="desc">{d}</td></tr>').format(t=term_fn(term), d=desc)
    return html + '</table>'


def _shortcut_table(rows):
    """ Two-column table of (keys, description); keys rendered as keycaps. """
    return _two_col_table(rows, _key)


def _defn_table(rows):
    """ Two-column table of (term, description); term rendered bold. """
    return _two_col_table(rows, lambda t: '<b>{:s}</b>'.format(t))


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

        "<h3>Next: measure a meteor</h3>"
        "<p>With a good platepar, switch to <b>Manual Reduction</b> (button under the image) to "
        "measure a meteor frame by frame &ndash; see <a href=\"topic:mr_overview\">Manual "
        "reduction</a> (and <a href=\"topic:mr_fireballs\">measuring fireballs</a>).</p>"
        + _nav_links(next_pair=('astrometry', 'Calibrate astrometry'),
                     related=[('residuals', 'Checking the fit'), ('mr_overview', 'Manual reduction')])
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
        "any image. Toggle it with the <b>Auto Levels</b> button at the top of the tab or with "
        + _key(c + " + A") + "; toggling again returns to your manual levels. While auto is on, "
        "the handles are locked.</p>"
        + _callout("Levels are display-only. Set them so you can comfortably see the stars you need "
                   "to pick &ndash; they have no effect on the calibration result.")
        + _nav_links(related=[('tabs', 'Guide to the tabs')])
    )
    return _page("Levels (display contrast)", body)


def _topic_inputs(gui):
    body = (
        "<p class=\"lead\">SkyFit opens by asking for an input &ndash; point it at a folder or a "
        "file. Most types also need a <b>config</b> (use <b>-c .</b> to read the .config in the data "
        "folder) and a <b>platepar</b> for sky coordinates.</p>"

        "<h3>FF files (RMS)</h3>"
        "<p>Compressed RMS frame blocks (maxpixel / avepixel / stdpixel / avgpixel over ~256 "
        "frames). Point at the <b>night folder</b>. Time and frame rate come from the FF file names "
        "and the config; both maxpixel and avepixel are available (" + _key("M") + " to toggle).</p>"

        "<h3>FR files (RMS)</h3>"
        "<p>Fast-read cut-outs holding the bright pixels of each detection. They load "
        "<b>alongside FF</b> files &ndash; add <b>-r / --fr</b>. In Manual Reduction, step through FR "
        "lines with " + _key(", / .") + ".</p>"

        "<h3>Image sequences</h3>"
        "<p>Folders of PNG / JPG / BMP / TIFF / FITS (and raw .NEF / .CR2 if <i>rawpy</i> is "
        "installed). Name files <b>YYYYMMDD_hhmmss[.uuuuuu]</b> so the time can be read, or pass "
        "<b>-t / --timebeg</b>; give the frame rate with <b>-f / --fps</b>. A single image opens in "
        "<b>single-image mode</b> (no frame stepping).</p>"

        "<h3>Video files</h3>"
        "<p>mp4 / avi / etc. The frame rate is read from the file; the <b>start time</b> is taken "
        "from the file name (YYYYMMDD_hhmmss) or must be given with <b>-t / --timebeg</b> &ndash; "
        "SkyFit errors if neither is available.</p>"

        "<h3>UWO .vid</h3>"
        "<p>University of Western Ontario EMCCD / CAMO / ASGARD <b>.vid</b> files. Each frame carries "
        "an <b>embedded timestamp</b>, so no start time or frame rate needs to be supplied &ndash; "
        "they are read straight from the file.</p>"

        "<h3>DFN fireball stills</h3>"
        "<p>Desert Fireball Network long-exposure stills (raw .NEF / .CR2). Timing is encoded as a "
        "<b>de Bruijn</b> sequence of shutter breaks &ndash; recover it in the "
        "<a href=\"topic:debruijn\">Debruijn</a> tab. The maxpixel/avepixel toggle is disabled for "
        "this single-exposure format.</p>"

        "<h3>Useful options</h3>"
        "<ul>"
        "<li><b>-c . / --config</b> &ndash; read the config in the data folder.</li>"
        "<li><b>-g / --gamma</b> &ndash; camera gamma (vital for photometry).</li>"
        "<li><b>--flipud</b> &ndash; flip images / videos upside down.</li>"
        "<li><b>--expratio</b> &ndash; exposure ratio for shutter-chopped long exposures.</li>"
        "<li><b>-m / --mask</b> &ndash; apply a mask.</li>"
        "</ul>"
        + _nav_links(related=[('tabs', 'Guide to the tabs'), ('frfiles', 'FR files')])
    )
    return _page("Data input types", body)


def _topic_tabs(gui):
    mode = _mode(gui)
    mode_name = "SkyFit" if mode == 'skyfit' else "Manual Reduction"

    rows = []
    rows.append(("Levels",
                 "Adjust image brightness and contrast with the histogram &ndash; drag the handles, "
                 "or press " + _key(_ctrl(gui) + " + A") + " for auto levels."))

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
                 "Draw or paint a mask over obstructions (roofs, trees) so the star and meteor "
                 "detectors ignore them."))
    rows.append(("Settings",
                 "Display options: which overlays to show (catalog / detected / selected stars, "
                 "constellations, coordinate grids, distortion), image gamma, magnitude limits, "
                 "invert colours and more."))

    if mode == 'manualreduction' and _is_dfn(gui):
        rows.append(("Debruijn",
                     "Recover the time of a DFN fireball from its shutter-break sequence."))

    rows.append(("Help", "This guide."))

    # Link each tab name to its dedicated help page where one exists
    tab_topics = {
        "Levels": "levels", "Fit Parameters": "astrometry", "Station": "station",
        "Star Detection": "stardetect", "Mask": "mask", "Settings": "settings",
        "Debruijn": "debruijn",
    }
    linked_rows = []
    for name, desc in rows:
        if name in tab_topics:
            name = '<a href="topic:{0}">{1}</a>'.format(tab_topics[name], name)
        linked_rows.append((name, desc))

    body = (
        "<p class=\"lead\">The tabs run down the right-hand edge of the window. Click a tab to open "
        "it, and click it again to collapse the panel. Tab names below link to their help pages. "
        "Which tabs appear depends on the mode (currently <b>" + mode_name + "</b>).</p>"
        + _defn_table(linked_rows)
    )
    return _page("Guide to the tabs", body)


def _topic_astrometry(gui):
    c = _ctrl(gui)
    body = (
        "<p>The goal is to pair at least <b>14 catalog stars</b> spread uniformly across the whole "
        "image (more is better, and include some near the horizon). Then fit.</p>"
        "<p><b>Find Best Frame</b> picks a good image to calibrate on: it scores every frame of "
        "the night on star distribution and quality, and on sky condition &ndash; the darkest, "
        "most uniform background and sharpest stars, ranked against the rest of the night &ndash; "
        "so moonlit or hazy frames lose to clean ones even when they show plenty of stars.</p>"
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
        "<h3>Running the fit steps independently</h3>"
        "<p>Auto Fit chains three steps that can also be run one at a time from the Fit Parameters "
        "tab, each gated on its prerequisites:</p>"
        "<ul>"
        "<li><b>Find Pairs</b> &ndash; matches the detected stars to catalog stars projected "
        "through the <i>current</i> platepar (nearest neighbour within a radius scaled to the star "
        "size). Replaces your current pairs and does not fit anything, so it needs a reasonably "
        "good platepar to start from.</li>"
        "<li><b>Fit</b> &ndash; fits the platepar to the current pairs (manual picks or found "
        "pairs alike). Needs enough pairs.</li>"
        "<li><b>Residuals</b> &ndash; recomputes the residuals of the current pairs against the "
        "current platepar <i>without fitting</i>. Use it after manually adjusting the platepar, or "
        "right after Find Pairs to judge the calibration before committing to a fit.</li>"
        "</ul>"
        "<h3>Validate Across Frames</h3>"
        "<p>Checks how well the fit generalizes beyond the calibration frame: detected stars "
        "(CALSTARS) from a coverage-selected subset of the night are matched to a catalog 1.5 "
        "mag deeper than the display, with the corner cells filled whenever the dataset has "
        "stars there. Only the pointing is refit per frame (distortion and scale frozen), so "
        "mount drift over the night is separated from distortion error and reported on its own. "
        "Pairings are cleaned the same way the auto fit cleans them (blend and photometric "
        "rejection), and match failures are classified: <b>recurring stars</b> (same sky star "
        "failing on many frames &ndash; a catalog gap, double or variable), <b>transient</b> "
        "(moving objects), and <b>persistent-in-place</b> &ndash; the last, sitting on hot map "
        "cells, marks real misfit beyond the match radius. A rising residual toward large radii "
        "means the distortion model does not generalize to the corners.</p>"
        "<p><b>Refit W/ Night</b> then feeds the validated pairs back into the astrometric fit. "
        "The pairs stay grouped by source frame and the multi-image fit projects each frame's "
        "catalog stars at that frame's own time, so picks from different times combine exactly. "
        "The set is spatially balanced so the star-rich centre does not dominate (corner pairs "
        "are kept in full), and each frame's measured pointing drift is compensated so the "
        "single fitted pointing is consistent with every frame. The refit is then validated "
        "against the night and only kept if it improves on the current platepar, so a refit "
        "never makes the calibration worse. This typically improves the corners substantially "
        "when the calibration frame had few corner stars; on an already well-generalizing "
        "platepar it simply reports that no improvement was found. Photometry is untouched "
        "&ndash; it must come from a single frame.</p>"
        "<p>Press " + _key("L") + " for the astrometry residual plot.</p>"
        + "<h3>Relevant shortcuts</h3>"
        + _shortcut_table([
            ("LEFT CLICK", "Centroid the star under the cursor"),
            (c + " + LEFT CLICK", "Manual (forced) star position"),
            ("ENTER / SPACE", "Accept the star pair"),
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
        "<p class=\"lead\">Photometry converts pixel intensity into stellar magnitude, which is "
        "needed to estimate meteor brightness and mass. It is fitted <b>automatically</b> from your "
        "calibration stars; press " + _key("P") + " (or the " + _btn("Photometry") + " button) to "
        "open the plot and review/tune it.</p>"

        "<h3>How magnitude is measured</h3>"
        "<p>A star's <b>flux</b> is the sum of its pixel values above the background. Magnitude is "
        "<b>m = -2.5 &middot; log10(flux) + ZP</b>. The log turns brightness <i>ratios</i> into "
        "additive steps; the constant <b>2.5</b> comes from the historical scale where 5 magnitudes "
        "= a factor of 100 in brightness (and 100<sup>1/5</sup> &asymp; 2.512); the <b>minus</b> "
        "keeps the old convention that <i>brighter stars have smaller magnitudes</i>. <b>ZP</b> (zero "
        "point) ties the instrument to the catalog scale.</p>"
        "<p><b>LSP</b> = <i>log sum pixel</i> = log10(sum of the star's pixel values) &ndash; the log "
        "of the measured flux. SkyFit fits magnitude as a straight line in LSP.</p>"

        "<h3>Left plot &ndash; catalog vs uncalibrated magnitude</h3>"
        "<p>Each star is plotted as catalog magnitude (vertical) against its <b>uncalibrated "
        "magnitude</b>, -2.5 &middot; LSP (horizontal). <b>Red</b> points are raw (extinction "
        "corrected); <b>blue</b> points are also corrected for vignetting. The line is the fit "
        "(<i>slope &middot; LSP + offset</i>).</p>"
        + _callout("Both axes are in magnitudes, so a good fit lies on the <b>1:1 diagonal</b> "
                   "(slope 1). A slope that departs from 1 means the camera <b>gamma</b> is wrong "
                   "&ndash; fix gamma until the points fall on the 1:1 line. Saturated stars are "
                   "circled and excluded from the fit.")

        + "<h3>Right insets (top to bottom)</h3>"
        "<p><b>1. Vignetting &mdash; residual vs radius.</b> Fit residual against distance from the "
        "image centre. After correction the points should sit <b>flat around zero</b>; the dotted "
        "curve is the vignetting model (light loss toward the edges) and should follow the measured "
        "drop-off. Fitted only when no flat field is loaded.</p>"
        "<p><b>2. Extinction &mdash; residual vs elevation.</b> Residual (with extinction removed) "
        "against star elevation; the dotted curve is the atmospheric extinction model (stars dim "
        "near the horizon). The single <b>extinction scale</b> knob scales the <i>total</i> modelled "
        "loss; adjust it so the model follows the points and the residuals are flat. In practice it "
        "is only really tunable at <b>very low elevations</b> &ndash; near the zenith the extinction "
        "is tiny, so points up there carry almost no information about it.</p>"
        + _callout("The extinction model is deliberately simple and only <b>first-order</b> &ndash; "
                   "not very accurate. It sums three atmospheric components &ndash; <b>Rayleigh "
                   "scattering</b>, <b>aerosol scattering</b> and <b>ozone</b> &ndash; scaled by air "
                   "mass (Green 1992, assuming the human eye), all relative to the zenith. The "
                   "extinction scale multiplies that single combined total rather than tuning each "
                   "component, which may not be the optimal correction.", "note")
        + "<p><b>3. Residuals, S/N &amp; limiting magnitude.</b> Fit residual against catalog "
        "magnitude, each point coloured by its <b>signal-to-noise ratio (S/N)</b> (colour bar "
        "0&ndash;15, with marks at S/N 5 and 10). A star's S/N is its flux divided by the noise "
        "(background + photon noise) in its aperture &ndash; high S/N is a confident measurement.</p>"
        "<p>The dashed vertical lines mark the <b>limiting magnitude (LM)</b>. SkyFit fits "
        "log10(S/N) against calibrated magnitude and reads off the magnitude where S/N falls to a "
        "target (5 and 10): the faintest star you can reliably measure. A <b>fainter LM</b> means a "
        "more sensitive setup; the printed equation lets you compute the LM for any S/N.</p>"
        "<p>On the image itself, each fitted star also gets a colour-coded ring (its residual) and "
        "an S/N label.</p>"

        "<h3>Tuning</h3>"
        "<ol>"
        "<li>Set the camera <b>gamma</b> correctly (science cameras 1.0, consumer ~0.45) so the left "
        "plot sits on the 1:1 line.</li>"
        "<li>Adjust the <b>extinction scale</b> (typically 0.6-1.0) so the extinction inset is "
        "flat.</li>"
        "<li>Only enable <b>fixed vignetting</b> for cameras with well-measured vignetting.</li>"
        "<li>Remove saturated / high-residual stars and re-fit until the scatter is small "
        "(aim for &lt;= ~0.2 mag). For saturated-object work, launch with <b>--nobg</b> or "
        "<b>--peribg</b>.</li>"
        "</ol>"
        + "<h3>Relevant shortcuts</h3>"
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

        "<p>The colour-coded RMSD under \"Residuals:\" summarises the fit, and the small "
        "<b>Round-trip max</b> line beneath it shows the worst disagreement between the forward "
        "and reverse mappings anywhere in the image (visualised by the error overlay in the "
        "Settings tab). The <b>Residuals</b> button recomputes the residuals against the current "
        "platepar without fitting &ndash; handy after manual platepar changes or after Find "
        "Pairs.</p>"

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
        + "<h3>Relevant shortcuts</h3>"
        + _shortcut_table([
            (c + " + D", "Load a dark frame"),
            (c + " + F", "Load a flat field"),
            (c + " + A", "Auto-adjust display levels"),
        ])
        + "<p>A <b>mask</b> hides parts of the image (e.g. obstructions) from the star catalog - "
        "see the <a href=\"topic:mask\">mask drawing</a> topic. Flat/dark are applied to the image "
        "before measurement; the <b>--flatbiassub</b> option also subtracts bias from the flat.</p>"
        + _callout("Never load the <b>auto-generated flat</b> that RMS produces (e.g. flat.bmp) into "
                   "SkyFit, and never use it operationally &ndash; it is not a true flat field and "
                   "will corrupt the photometry. Only load a <b>dedicated flat</b> made separately, "
                   "and only on <b>science-grade sensors</b>.", "note")
    )
    return _page("Calibration files (dark / flat / mask)", body)


def _topic_stardetect(gui):
    params = _defn_table([
        ("Adaptive gate factor <span class=\"tip\">(def. 3.0)</span>",
         "How far above the frame's own measured noise a peak must rise to count as a star, as a "
         "multiple of that noise. <b>Lower</b> detects fainter stars but also more noise; "
         "<b>higher</b> keeps only the bright, confident ones. The single most useful knob &ndash; "
         "tune it first. It is <b>per camera</b>: 3.0 suits most, but a dark, low-gain sensor whose "
         "averaged frame is dominated by fixed-pattern noise (hot pixels) floods at 3.0 and "
         "measures clean at 10&ndash;15. <b>Tune</b> sweeps it against the catalog and picks a "
         "value."),
        ("Neighborhood size <span class=\"tip\">(def. 10 px)</span>",
         "Size of the local window used to pick one peak per star. <b>Larger</b> merges close stars "
         "(fewer detections); <b>smaller</b> separates them but can split one bright star into "
         "several. Set it a little larger than your typical star spacing."),
        ("Max stars <span class=\"tip\">(Station Config, def. 400)</span>",
         "The <code>max_stars</code> value that <b>Save Config</b> writes to the station config. "
         "This bounds the star extraction cost of the nightly pipeline on the station, which only "
         "needs a modest sample to track calibration drift &ndash; <b>400 is recommended</b>."),
        ("Max stars (detection depth) <span class=\"tip\">(SkyFit Session, def. 800)</span>",
         "Candidate budget used by <b>Re-Detect in this session only</b> &ndash; it is never saved "
         "to the config. Initial plate fitting benefits from a deep, frame-wide star sample, so feel "
         "free to raise it. When more candidates are found than the budget, they are subsampled "
         "evenly across the frame (most prominent first within each region), not simply brightest "
         "first."),
        ("Max global intensity <span class=\"tip\">(def. 230)</span>",
         "Median image level (8-bit scale) above which a frame is considered too bright to contain "
         "stars and is skipped entirely. Raise it if twilight or moonlit frames that still show "
         "stars are being rejected; frames near saturation are never worth processing."),
        ("Gamma <span class=\"tip\">(def. 1.0)</span>",
         "Camera gamma used when measuring stars (not the display gamma). Values below 1 lift "
         "faint stars out of the background so they get detected. Saved to the config "
         "<code>[Capture]</code> section and also stored in the platepar for photometry."),
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

        "<p>The parameters are split into two groups: <b>Station Config</b> values are what "
        "<b>Save Config</b> writes to the station config file (they control the nightly pipeline), "
        "while <b>SkyFit Session Only</b> values are used by re-detection here and are never "
        "saved.</p>"

        "<h3>How to use it</h3>"
        "<ol>"
        "<li>Adjust the parameters below and click <b>Redetect</b> to re-run detection on the "
        "current image (or <b>Redetect all</b> for every image). Detected stars are drawn on the "
        "image so you can see the effect.</li>"
        "<li>Click <b>Tune</b> to let SkyFit balance the parameters and catalog limiting magnitude "
        "so the number of detected stars roughly matches the catalog stars in the field.</li>"
        "<li>Tick <b>Use Override Detections</b> to feed these detections into fitting and Auto "
        "Fit instead of CALSTARS.</li>"
        "<li>Tick <b>Ignore CALSTARS</b> if the CALSTARS detections themselves are the problem. "
        "Without it, the night-wide steps (Validate Across Frames, Refit W/ Night, Find Best "
        "Frame, Save CALSTARS) substitute your detections <i>per frame</i> and keep the CALSTARS "
        "entry for every frame you did not re-detect &ndash; and since Re-Detect All only reaches "
        "the FF files on disk while CALSTARS usually covers the whole night, most of the pool can "
        "stay as it was. With it ticked those frames are left out entirely. <b>Re-Detect All ticks "
        "it for you</b>; untick it to go back to merging.</li>"
        "<li><b>Save Config</b> writes the parameters to your config file so the station pipeline "
        "reuses them. Note it saves <b>Config max stars</b>, not the SkyFit session budget &ndash; "
        "deep detection is for calibration here, while the nightly pipeline should stay cheap.</li>"
        "<li><b>Reset to Defaults</b> returns every slider in this tab to the recommended "
        "values.</li>"
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
        "<p class=\"lead\">The mask marks parts of the image to <b>ignore</b>. Its main job is for "
        "<b>detection</b>: the star and meteor detectors skip masked pixels, so obstructions don't "
        "produce false detections or get confused for stars and meteors. (It also keeps catalog "
        "stars over those areas out of the fit.)</p>"

        "<h3>What to mask</h3>"
        "<p>Mask anything that is <b>not clear sky</b> and could trigger or corrupt detections: "
        "rooflines, trees, poles, wires, the horizon, timestamps/illuminated overlays, persistent "
        "lights, and any fixed bright reflections.</p>"

        "<h3>How carefully to mask</h3>"
        "<p>It does not need to be pixel-perfect, but don't be sloppy either. Aim to <b>maximise the "
        "observable sky</b> (mask as little as possible) while still fully covering each "
        "obstruction. Leave a small margin of about <b>5&ndash;10 px</b> around obstructions so the "
        "mask still holds if the camera shifts slightly. Too tight and a small camera move exposes "
        "the obstruction; too loose and you throw away usable sky.</p>"

        "<h3>Tools</h3>"
        "<ul>"
        "<li><b>Draw mode</b>: click to lay down polygon vertices; the enclosed area is masked.</li>"
        "<li><b>Brush mode</b>: paint / erase the mask freehand; adjust the brush size.</li>"
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
        ("Show Round-Trip Error Overlay", "Heatmap of the disagreement (px) between the forward "
         "and reverse astrometric mappings across the image. Bright areas mark where the two "
         "distortion polynomials are inconsistent, so the catalog overlay cannot be trusted there. "
         "The <b>Threshold</b> slider sets the error below which the overlay is fully transparent; "
         "the maximum value is always shown under the residuals in the Fit Parameters tab."),
        ("Invert Colors", "Invert the image (dark stars on a light background)."),
        ("Single Click Photometry", "Measure photometry with a single click while picking."),
    ]

    catalog_rows = [
        ("Gamma", "Display gamma of the image (visual only &ndash; not the camera gamma)."),
        ("Lim Mag", "Catalog limiting magnitude: how faint to load catalog stars."),
        ("Correct Mag for Ext./Vign.", "Display only: applies the extinction and vignetting "
         "correction to the catalog magnitudes so you can visualise vignetting and extinction. "
         "It does not change the fit."),
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
        "<p class=\"lead\">Satellite tracks overlay the predicted paths of satellites on the image, "
        "so you can identify a satellite trail or tell one apart from a meteor. Needs the "
        "<b>skyfield</b> package (the feature is disabled with a warning if it is missing) and a "
        "valid platepar to project positions onto the image.</p>"

        "<h3>How positions are predicted</h3>"
        "<p>Each satellite's orbit is propagated from its <b>TLE</b> (two-line element set) with the "
        "SGP4 model via skyfield, converted to RA/Dec for your station and time, then projected to "
        "image pixels through the platepar. Only satellites that are <b>sunlit</b> (not in Earth's "
        "shadow) are shown.</p>"

        "<h3>What you see</h3>"
        "<ul>"
        "<li>A coloured <b>track line</b> spanning the clip's time range &ndash; one per satellite.</li>"
        "<li><b>Direction arrows</b> at the start, middle and end showing the direction of motion.</li>"
        "<li>The satellite <b>name label</b>.</li>"
        "</ul>"

        "<h3>Manual Reduction: matching a moving satellite</h3>"
        "<p>In Manual Reduction mode a <b>marker</b> shows the satellite's predicted position at the "
        "<i>current frame's time</i>, interpolated along the track. As you step through frames it "
        "advances, so you can follow a satellite frame by frame and match it to a trail. Turn on "
        "<b>Automatically compute tracks</b> to refresh on every frame change (off by default "
        "because it is slow); otherwise press <b>Redraw Satellite Tracks</b> after changing frame or "
        "TLEs.</p>"
        "<p>Click a satellite's <b>name label</b> to open its page on n2yo.com "
        "(<i>n2yo.com/satellite/?s=&lt;NORAD&nbsp;ID&gt;</i>) in your browser for orbit and pass "
        "details.</p>"

        "<h3>TLEs and getting a good match</h3>"
        "<p>SkyFit chooses TLEs in this order: a file or directory you pass with <b>--tle_file</b> "
        "or the <b>Load TLE File</b> button; otherwise TLEs downloaded from Celestrak's active "
        "catalog and cached locally (refreshed about daily &ndash; needs internet the first time). "
        "Point it at a <b>directory</b> of dated TLE files (named <i>TLE_YYYYMMDD_HHMMSS_...</i>) and "
        "it picks the one closest in time to your observation. <b>Reset TLE Selection</b> returns to "
        "the auto-downloaded set.</p>"
        + _callout("For the best match, use a TLE set from the <b>closest date to your data</b>. "
                   "When SkyFit opens it prints the chosen TLE file and its time offset in the "
                   "terminal (e.g. <i>\"Best file found ... (diff: X hours)\"</i>). If the TLEs are "
                   "more than a <b>few days</b> from the observation the predicted positions will "
                   "<b>not match</b> &ndash; orbits drift, so old TLEs are unreliable.", "note")

        + "<h3>Controls</h3>"
        + _shortcut_table([
            (c + " + T", "Toggle satellite tracks"),
        ])
        + "<ul>"
        "<li><b>Show Satellite Tracks</b> / <b>Automatically compute tracks</b> &ndash; in the "
        "Settings tab.</li>"
        "<li><b>Load TLE File</b> &ndash; pick a local TLE file or a directory of dated TLE files.</li>"
        "<li><b>Redraw Satellite Tracks</b> &ndash; recompute now (after loading new TLEs).</li>"
        "<li>Command line: <b>--sattracks</b> to enable, <b>--tle_file PATH</b> for a local set.</li>"
        "</ul>"
        + _nav_links(related=[('settings', 'Settings tab')])
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
        (c + " + /", "Open this keyboard reference"),
        ("SHIFT + F1", "Open the help guide"),
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
        (c + " + X", "astrometry.net (image upload)"),
        (c + " + SHIFT + X", "astrometry.net (star XY only)"),
        (c + " + SHIFT + B", "Fit spectral bands <span class=\"tip\">(experimental)</span>"),
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
                     related=[('mr_fireballs', 'Measuring fireballs'), ('mr_lightcurve', 'Light curve & saving')])
    )
    return _page("Overview &amp; quick start", body)


def _topic_mr_picking(gui):
    c = _ctrl(gui)
    body = (
        "<p class=\"lead\">Enter picking mode with " + _key(c + " + R") + ", then mark the meteor on "
        "each frame in time order, stepping with " + _key("Left / Right") + ".</p>"

        "<h3>Centroid vs manual pick</h3>"
        "<ul>"
        "<li><b>Left click</b> &ndash; <b>centroid</b>: snaps to the intensity-weighted centre "
        "inside the aperture. Use it whenever the segment is a clean blob.</li>"
        "<li>" + _key(c + " + Left click") + " &ndash; <b>manual (forced) pick</b> at the exact "
        "cursor position, with no centroiding. Use it when a centroid would be dragged off by a "
        "neighbour, wake, or a saturated / elongated shape.</li>"
        "<li><b>Right click</b> removes the pick on the current frame.</li>"
        "</ul>"

        "<h3>Aperture</h3>"
        "<p>Hold " + _key(c + " + Scroll") + " to resize the circular aperture. Match it to the "
        "segment: big enough to enclose all of its light, small enough to exclude neighbours and "
        "background. The aperture also sets the photometry region and, for saturated blobs, doubles "
        "as a positioning aid (see <a href=\"topic:mr_fireballs\">fireballs</a>).</p>"

        "<h3>Slow vs fast meteors</h3>"
        "<ul>"
        "<li><b>Slow</b> (short, roughly round segment each frame): just <b>centroid</b> (left "
        "click) &ndash; the centre is well defined.</li>"
        "<li><b>Fast</b> (the per-frame segment is an elongated streak): the centroid is ambiguous "
        "along the streak, so measure the <b>leading edge</b> (the front of the streak, in the "
        "direction of travel) <i>consistently</i> on every frame with a "
        + _key(c + " + Left click") + ". For streaked events consider "
        "<a href=\"topic:mr_astra\">ASTRA</a>, which is built for them.</li>"
        "</ul>"

        "<h3>The great-circle fit as a guide</h3>"
        "<p>Turn on <b>Show Great Circle Line</b> (Settings). A meteor follows a great circle on the "
        "sky, so SkyFit fits one through your picks and draws it as a <b>purple dotted arc</b>. Where "
        "a frame is faint or ambiguous, place the pick <b>on the arc</b> &ndash; it shows the path "
        "the meteor must follow and exposes any pick that sits off the line.</p>"

        + "<h3>Relevant shortcuts</h3>"
        + _shortcut_table([
            ("LEFT CLICK", "Centroid the meteor at the cursor"),
            (c + " + LEFT CLICK", "Force a pick at the exact cursor position"),
            (c + " + SCROLL", "Resize the aperture"),
            ("RIGHT CLICK", "Remove the pick on this frame"),
            ("ALT / Num0 + LEFT CLICK", "Mark a gap (DFN sequences)"),
            ("Left / Right", "Previous / next frame"),
            (c + " + Left / Right", "+/- 10 frames"),
            ("Down / Up", "+/- 25 frames"),
            (", / .", "Previous / next FR line"),
            ("M", "Show maxpixel"),
            ("K", "Subtract average"),
            ("T", "Toggle refraction correction"),
        ])
        + _nav_links(next_pair=('mr_lightcurve', 'Light curve & saving'),
                     related=[('mr_fireballs', 'Measuring fireballs'), ('mr_astra', 'ASTRA')])
    )
    return _page("Picking meteor positions", body)


def _topic_fireballs(gui):
    c = _ctrl(gui)
    body = (
        "<p class=\"lead\">Fireballs are much harder to measure than ordinary meteors: they "
        "<b>saturate</b>, develop a trailing <b>wake</b>, and can <b>fragment</b>, so the bright "
        "blob no longer has a clean, well-defined centre.</p>"

        "<h3>Strategy</h3>"
        "<ol>"
        "<li><b>Measure the easy parts first.</b> Where the fireball is <i>not</i> saturating and "
        "looks like a clean point source &ndash; usually the <b>beginning and the end</b> &ndash; "
        "centroid normally. These good picks anchor the trajectory.</li>"
        "<li><b>Let the great circle guide the middle.</b> With the clean picks in place, the "
        "<a href=\"topic:mr_picking\">great-circle fit</a> (purple arc) shows the path through the "
        "difficult middle frames &ndash; place picks along it.</li>"
        "<li><b>Saturated-blob frames.</b> When the fireball saturates into a big blob the centroid "
        "is meaningless. <b>Size the aperture to match the blob</b> (" + _key(c + " + Scroll") + "), "
        "then make a <b>manual pick</b> (" + _key(c + " + Left click") + ") <b>on the great-circle "
        "line</b>, using the aperture circle to judge the blob's centre across-track.</li>"
        "</ol>"

        + _callout("Along-track (in-track) position on saturated frames is the hard part and can't "
                   "be read precisely. A fireball moves smoothly and never jumps back and forth, so "
                   "aim for <b>roughly even spacing between consecutive frames</b>, letting the "
                   "spacing of the clean frames before and after guide where the middle picks fall.")

        + "<p>Wake and fragmentation also pull a centroid backwards or sideways, so prefer manual "
        "leading-edge picks there as well. <a href=\"topic:mr_astra\">ASTRA</a> can help refine "
        "difficult picks.</p>"
        + _nav_links(related=[('mr_picking', 'Pick meteor positions'), ('mr_astra', 'ASTRA')])
    )
    return _page("Measuring fireballs", body)


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
        "<p>Reach for ASTRA when picking by hand is unreliable: <b>fast / streaked</b> meteors where "
        "you'd otherwise pick the leading edge frame by frame, EMCCD-style data, or to refine the "
        "difficult middle frames of a <a href=\"topic:mr_fireballs\">fireball</a>.</p>"

        "<h3>Getting started: the seed picks</h3>"
        "<p>ASTRA needs a few manual leading-edge picks (or load them from an ECSV/txt file) to "
        "start:</p>"
        "<ol>"
        "<li><b>Two edge picks</b> &ndash; one on the <b>first</b> and one on the <b>last</b> frame "
        "of the event. These set the <b>frame range</b> ASTRA will process.</li>"
        "<li><b>Three more picks</b> on frame-adjacent frames somewhere in a <b>good-SNR</b> section "
        "in between. These <b>kickstart the fit</b> (seed the trajectory and velocity).</li>"
        "</ol>"
        "<p>Make all of them on the <b>leading edge</b>. The READY / NOT READY indicators show when "
        "enough picks are present; hover over any parameter or icon for its tooltip.</p>"

        "<h3>Pick position &amp; photometry</h3>"
        "<p>By default ASTRA places its picks on the <b>leading edge</b> of the streak. To change "
        "that, set <b>pick_offset</b> (ASTRA parameter settings) to <i>center</i>, or to a custom "
        "float (in multiples of the streak-length standard deviation) to slide the pick along the "
        "streak axis.</p>"
        "<p>ASTRA also does <b>automated photometry</b>. The <b>photom_thresh</b> parameter "
        "(luminosity threshold as a fraction of the peak) sets the photometric mask: <b>raise</b> it "
        "to tighten the mask to only the brightest pixels, <b>lower</b> it to expand the mask to "
        "include fainter wings.</p>"
        + _nav_links(related=[('mr_picking', 'Pick meteor positions'),
                              ('mr_fireballs', 'Measuring fireballs')])
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
        (c + " + /", "Open this keyboard reference"),
        ("SHIFT + F1", "Open the help guide"),
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
# Section headers used to group topics on the home page, in display order.
# The two workflow sections are shown on the home in BOTH modes (so meteor measurement is
# discoverable while calibrating, and vice versa); the rest are filtered to the current mode.
SECTION_ORDER = ["Getting started", "Calibration", "Meteor measurement", "Tools & tabs", "Reference"]
CROSS_MODE_SECTIONS = {"Calibration", "Meteor measurement"}

HELP_TOPICS = [
    # SkyFit
    ('overview',          dict(title="Overview &amp; quick start",        modes=('skyfit',),          enabled=_always,        build=_topic_overview,        section="Getting started",
                               desc="Start here: what SkyFit does and the fastest way to calibrate.")),
    ('inputs',            dict(title="Data input types",                  modes=('skyfit',),          enabled=_always,        build=_topic_inputs,          section="Getting started",
                               desc="What data SkyFit can load and what each needs.")),
    ('astrometry',        dict(title="Calibrate astrometry (pick stars)", modes=('skyfit',),          enabled=_always,        build=_topic_astrometry,      section="Calibration",
                               desc="Pick stars and fit the plate by hand.")),
    ('photometry',        dict(title="Photometry",                        modes=('skyfit',),          enabled=_always,        build=_topic_photometry,      section="Calibration",
                               desc="Calibrate brightness: extinction, vignetting, gamma.")),
    ('residuals',         dict(title="Checking the fit (residual plots)",  modes=('skyfit',),          enabled=_always,        build=_topic_residuals,       section="Calibration",
                               desc="Read the residual plots and the values to hit.")),
    ('calibration_files', dict(title="Calibration files (dark/flat/mask)",modes=('skyfit',),          enabled=_always,        build=_topic_calibration_files, section="Calibration",
                               desc="Load dark, flat and mask frames.")),
    ('station',           dict(title="Station &amp; location",            modes=('skyfit',),          enabled=_always,        build=_topic_station,         section="Calibration",
                               desc="Observer location, station moves, auto-refit.")),
    ('geopoints',         dict(title="Geo points (ground references)",    modes=('skyfit',),          enabled=_has_geopoints, build=_topic_geopoints,       section="Calibration",
                               desc="Calibrate pointing from terrestrial landmarks.")),
    ('tabs',              dict(title="Guide to the tabs",                 modes=('skyfit',),          enabled=_always,        build=_topic_tabs,            section="Tools & tabs",
                               desc="What each tab on the right does.")),
    ('levels',            dict(title="Levels (display contrast)",         modes=('skyfit',),          enabled=_always,        build=_topic_levels,          section="Tools & tabs",
                               desc="The histogram, black/white points, auto levels.")),
    ('stardetect',        dict(title="Star detection override",           modes=('skyfit',),          enabled=_always,        build=_topic_stardetect,      section="Tools & tabs",
                               desc="Re-detect stars with tunable parameters.")),
    ('mask',              dict(title="Mask drawing",                      modes=('skyfit',),          enabled=_always,        build=_topic_mask,            section="Tools & tabs",
                               desc="Ignore obstructions so detection isn't fooled.")),
    ('sattracks',         dict(title="Satellite tracks",                  modes=('skyfit',),          enabled=_always,        build=_topic_sattracks,       section="Tools & tabs",
                               desc="Overlay predicted satellite passes (toggle in Settings).")),
    ('frfiles',           dict(title="FR files",                          modes=('skyfit',),          enabled=_has_fr,        build=_topic_frfiles,         section="Tools & tabs",
                               desc="Work with fast-read meteor detection files.")),
    ('settings',          dict(title="Settings tab",                      modes=('skyfit',),          enabled=_always,        build=_topic_settings,        section="Tools & tabs",
                               desc="Every option in the Settings tab explained.")),
    ('shortcuts_skyfit',  dict(title="Keyboard reference",                modes=('skyfit',),          enabled=_always,        build=_topic_shortcuts_skyfit, section="Reference",
                               desc="Every keyboard shortcut, grouped.")),

    # Manual Reduction
    ('mr_overview',       dict(title="Overview &amp; quick start",        modes=('manualreduction',), enabled=_always,        build=_topic_mr_overview,     section="Getting started",
                               desc="Start here: measure a meteor frame by frame.")),
    ('mr_inputs',         dict(title="Data input types",                  modes=('manualreduction',), enabled=_always,        build=_topic_inputs,          section="Getting started",
                               desc="What data SkyFit can load and what each needs.")),
    ('mr_picking',        dict(title="Pick meteor positions",             modes=('manualreduction',), enabled=_always,        build=_topic_mr_picking,      section="Meteor measurement",
                               desc="Mark the meteor position on each frame.")),
    ('mr_fireballs',      dict(title="Measuring fireballs",                modes=('manualreduction',), enabled=_always,        build=_topic_fireballs,       section="Meteor measurement",
                               desc="Saturation, wake, fragmentation: how to pick them.")),
    ('mr_lightcurve',     dict(title="Light curve &amp; saving",          modes=('manualreduction',), enabled=_always,        build=_topic_mr_lightcurve,   section="Meteor measurement",
                               desc="View the light curve and export results.")),
    ('debruijn',          dict(title="DFN / Debruijn timing",             modes=('manualreduction',), enabled=_is_dfn,        build=_topic_debruijn,        section="Meteor measurement",
                               desc="Recover DFN fireball timing.")),
    ('mr_astra',          dict(title="ASTRA (automated picking)",         modes=('manualreduction',), enabled=_always,        build=_topic_mr_astra,        section="Meteor measurement",
                               desc="Automate or refine picks with ASTRA.")),
    ('mr_tabs',           dict(title="Guide to the tabs",                 modes=('manualreduction',), enabled=_always,        build=_topic_tabs,            section="Tools & tabs",
                               desc="What each tab on the right does.")),
    ('mr_levels',         dict(title="Levels (display contrast)",         modes=('manualreduction',), enabled=_always,        build=_topic_levels,          section="Tools & tabs",
                               desc="The histogram, black/white points, auto levels.")),
    ('mr_settings',       dict(title="Settings tab",                      modes=('manualreduction',), enabled=_always,        build=_topic_settings,        section="Tools & tabs",
                               desc="Every option in the Settings tab explained.")),
    ('shortcuts_mr',      dict(title="Keyboard reference",                modes=('manualreduction',), enabled=_always,        build=_topic_shortcuts_mr,    section="Reference",
                               desc="Every keyboard shortcut, grouped.")),
]

_TOPIC_MAP = dict(HELP_TOPICS)


def shortcutsTopicId(gui):
    """ Id of the keyboard reference topic for the mode the GUI is currently in. """
    return 'shortcuts_skyfit' if _mode(gui) == 'skyfit' else 'shortcuts_mr'


def _enabled_topics(gui, mode_filter=True):
    """ Return (id, meta) topics whose feature gate is satisfied. If mode_filter is True, also
        restrict to the current mode; otherwise return all enabled topics across modes. """
    mode = _mode(gui)
    out = []
    for topic_id, meta in HELP_TOPICS:
        if mode_filter and mode not in meta['modes']:
            continue
        try:
            if not meta['enabled'](gui):
                continue
        except Exception:
            continue
        out.append((topic_id, meta))
    return out


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

    def _row(topic_id, meta):
        return ('<tr><td valign="top"><a href="topic:{tid}">{title}</a></td>'
                '<td valign="top" class="desc">{desc}</td></tr>').format(
                    tid=topic_id, title=meta['title'], desc=meta.get('desc', ''))

    mode_topics = _enabled_topics(gui, mode_filter=True)   # current mode only
    all_topics = _enabled_topics(gui, mode_filter=False)   # across both modes

    # Search: flat list of matches across both modes (de-duplicated by title), no section headers
    if query:
        q = query.lower()
        seen_titles = set()
        matches = []
        for tid, m in all_topics:
            if q in m['title'].lower() or q in m.get('desc', '').lower():
                if m['title'] in seen_titles:
                    continue
                seen_titles.add(m['title'])
                matches.append((tid, m))
        body = "<h3>Search results</h3>"
        if not matches:
            body += "<p class=\"lead\">No topics match &ldquo;{:s}&rdquo;.</p>".format(query)
        else:
            body += ("<table cellspacing=\"0\" cellpadding=\"0\">"
                     + "".join(_row(tid, m) for tid, m in matches) + "</table>")
        return "<h2>SkyFit2 Help &mdash; {mode}</h2>{body}".format(mode=mode_name, body=body)

    # Normal home: group topics under section headers. Workflow sections (Calibration, Meteor
    # measurement) are shown regardless of mode so both workflows are discoverable; the rest are
    # filtered to the current mode.
    sections_html = ""
    for section in SECTION_ORDER:
        pool = all_topics if section in CROSS_MODE_SECTIONS else mode_topics
        rows = "".join(_row(tid, m) for tid, m in pool if m.get('section') == section)
        if rows:
            sections_html += ("<h3>" + section + "</h3>"
                              "<table cellspacing=\"0\" cellpadding=\"0\">" + rows + "</table>")

    # The keyboard reference is the most asked-for page, so pin it above the section list as well
    # as leaving it in its Reference section - at the bottom of a long list it was being missed.
    pinned = ("<p class=\"lead\"><b><a href=\"topic:" + shortcutsTopicId(gui) + "\">Keyboard "
              "reference</a></b> &ndash; every shortcut, grouped (" + _key(_ctrl(gui) + " + /")
              + ").</p>")

    body = (
        intro
        + pinned
        + sections_html
        + "<hr>"
        "<p class=\"lead\">Every tab has an <b>i</b> button in its top-right corner for help on "
        "that tab. Open this guide any time from the <b>Help</b> menu or <b>Shift+F1</b>; "
        + _key(_ctrl(gui) + " + /") + " jumps straight to the keyboard reference, and "
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
