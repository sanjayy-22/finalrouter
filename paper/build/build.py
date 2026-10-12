# -*- coding: utf-8 -*-
"""Rebuild word/document.xml of the IEEE template with the LLM-routing paper content.

usage: python3 build.py <unpacked_dir>
Keeps the original title-block structure (author table, section setup) and formatting
conventions; regenerates everything after the first section break.
"""
import re, sys
from xml.sax.saxutils import escape
from lxml import etree
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import content as C

D = sys.argv[1]
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ns = {"w": W}
path = f"{D}/word/document.xml"
raw = open(path, "rb").read()
tree = etree.fromstring(raw)
body = tree.find("w:body", ns)
els = list(body)

# ---------- citation numbering (order of first appearance) ----------
order = {}
def _collect(text):
    for grp in re.findall(r"\{([a-z0-9_,]+)\}", text):
        for k in grp.split(","):
            if k not in C.REFS:
                raise KeyError(k)
            order.setdefault(k, len(order) + 1)

for item in C.BODY:
    kind = item[0]
    if kind in ("P", "H1", "H2"):
        _collect(item[1])
    elif kind == "TABLE":
        for row in item[4]:
            for cell in row:
                _collect(cell)
unused = set(C.REFS) - set(order)
if unused:
    raise SystemExit(f"uncited references: {unused}")

def _fmt(nums):
    nums = sorted(nums)
    runs, start = [], nums[0]
    prev = start
    for n in nums[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n; continue
        runs.append((start, prev))
        if n is not None:
            start = prev = n
    parts = []
    for a, b in runs:
        if b - a >= 2:
            parts.append(f"[{a}]-[{b}]")
        elif b == a + 1:
            parts += [f"[{a}]", f"[{b}]"]
        else:
            parts.append(f"[{a}]")
    return ", ".join(parts)

def cite(text):
    return re.sub(r"\{([a-z0-9_,]+)\}", lambda m: _fmt([order[k] for k in m.group(1).split(",")]), text)

# ---------- XML helpers ----------
FONT = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>'
PLACEHOLDER = re.compile(r"(\[TBD\]|\[To be recorded\])")

def runs(text, sz, extra=""):
    out = []
    for seg in PLACEHOLDER.split(text):
        if not seg:
            continue
        hl = '<w:highlight w:val="yellow"/>' if PLACEHOLDER.fullmatch(seg) else ""
        out.append(f'<w:r><w:rPr>{FONT}{extra}{hl}<w:sz w:val="{sz}"/></w:rPr>'
                   f'<w:t xml:space="preserve">{escape(seg)}</w:t></w:r>')
    return "".join(out)

def p_body(text):
    return ('<w:p><w:pPr><w:keepNext w:val="0"/><w:spacing w:before="0" w:after="20" w:line="240" w:lineRule="auto"/>'
            '<w:ind w:firstLine="216"/><w:jc w:val="both"/></w:pPr>' + runs(cite(text), 20) + '</w:p>')

def p_h1(text):
    return ('<w:p><w:pPr><w:keepNext/><w:spacing w:before="100" w:after="40"/><w:jc w:val="center"/></w:pPr>'
            + runs(text, 20, '<w:b w:val="0"/><w:smallCaps/>') + '</w:p>')

def p_h2(text):
    return ('<w:p><w:pPr><w:keepNext/><w:spacing w:before="60" w:after="20"/><w:jc w:val="left"/></w:pPr>'
            + runs(text, 20, '<w:b w:val="0"/><w:i/>') + '</w:p>')

def p_eq(text, n):
    return ('<w:p><w:pPr><w:tabs><w:tab w:pos="2412" w:val="center"/><w:tab w:pos="4824" w:val="right"/></w:tabs>'
            '<w:spacing w:before="20" w:after="40" w:line="240" w:lineRule="auto"/></w:pPr>'
            f'<w:r><w:rPr>{FONT}<w:sz w:val="18"/></w:rPr><w:tab/><w:t xml:space="preserve">{escape(text)}</w:t>'
            f'<w:tab/><w:t>({n})</w:t></w:r></w:p>')

def sect(cols):
    return ('<w:p><w:pPr><w:sectPr><w:type w:val="continuous"/><w:pgSz w:w="11909" w:h="16834"/>'
            '<w:pgMar w:top="778" w:right="893" w:bottom="1440" w:left="893" w:header="288" w:footer="360" w:gutter="0"/>'
            f'<w:cols w:space="360" w:num="{cols}"/><w:docGrid w:linePitch="360"/></w:sectPr></w:pPr></w:p>')

def caption_table(num, title):
    return ('<w:p><w:pPr><w:keepNext/><w:spacing w:before="60" w:after="40"/><w:jc w:val="center"/></w:pPr>'
            f'<w:r><w:rPr>{FONT}<w:b/><w:sz w:val="16"/></w:rPr><w:t>TABLE {num}</w:t><w:br/>'
            f'<w:t xml:space="preserve">{escape(title)}</w:t></w:r></w:p>')

TC_MAR = ('<w:tcMar><w:top w:w="40" w:type="dxa"/><w:start w:w="50" w:type="dxa"/>'
          '<w:bottom w:w="40" w:type="dxa"/><w:end w:w="50" w:type="dxa"/></w:tcMar>')
BORDERS = ('<w:tblBorders>' + "".join(f'<w:{s} w:val="single" w:sz="4" w:color="808080"/>'
           for s in ("top", "left", "bottom", "right", "insideH", "insideV")) + '</w:tblBorders>')

def cell(text, w, header=False, align="left", sz=14):
    shd = '<w:shd w:val="clear" w:color="auto" w:fill="E7E6E6"/>' if header else ""
    extra = "<w:b/>" if header else '<w:b w:val="0"/>'
    return (f'<w:tc><w:tcPr><w:tcW w:type="dxa" w:w="{w}"/>{shd}{TC_MAR}<w:vAlign w:val="center"/></w:tcPr>'
            f'<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/>'
            f'<w:jc w:val="{"center" if header else align}"/></w:pPr>{runs(cite(text), sz, extra)}</w:p></w:tc>')

def table(headers, rows, widths, sz=14):
    total = sum(widths)
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    out = [f'<w:tbl><w:tblPr><w:tblW w:type="dxa" w:w="{total}"/><w:jc w:val="center"/>{BORDERS}'
           '<w:tblLayout w:type="fixed"/><w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" '
           'w:firstColumn="1" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr>'
           f'<w:tblGrid>{grid}</w:tblGrid>']
    out.append('<w:tr><w:trPr><w:tblHeader/></w:trPr>' +
               "".join(cell(h, w, header=True, sz=sz) for h, w in zip(headers, widths)) + '</w:tr>')
    for r in rows:
        out.append('<w:tr><w:trPr><w:cantSplit/></w:trPr>' +
                   "".join(cell(c, w, align="left" if (i == 0 or len(c) > 14) else "center", sz=sz)
                           for i, (c, w) in enumerate(zip(r, widths))) + '</w:tr>')
    out.append('</w:tbl>')
    return "".join(out)

def algo(title, lines):
    w = 4824
    paras = "".join(
        '<w:p><w:pPr><w:spacing w:before="0" w:after="10" w:line="240" w:lineRule="auto"/>'
        f'<w:ind w:left="{230 + (230 if l.startswith(tuple(f"{i})    " for i in range(1, 11))) else 0)}" w:hanging="230"/></w:pPr>'
        + runs(re.sub(r"^(\d+\))\s+", r"\1 ", l), 15) + '</w:p>' for l in lines)
    return ('<w:p><w:pPr><w:keepNext/><w:spacing w:before="60" w:after="20"/><w:jc w:val="center"/></w:pPr>'
            f'<w:r><w:rPr>{FONT}<w:b/><w:sz w:val="16"/></w:rPr><w:t xml:space="preserve">{escape(title)}</w:t></w:r></w:p>'
            f'<w:tbl><w:tblPr><w:tblW w:type="dxa" w:w="{w}"/><w:jc w:val="center"/>{BORDERS}'
            '<w:tblLayout w:type="fixed"/><w:tblLook w:val="04A0" w:firstRow="0" w:lastRow="0" w:firstColumn="0" '
            'w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr>'
            f'<w:tblGrid><w:gridCol w:w="{w}"/></w:tblGrid><w:tr><w:tc><w:tcPr><w:tcW w:type="dxa" w:w="{w}"/>'
            f'{TC_MAR}</w:tcPr>{paras}</w:tc></w:tr></w:tbl>'
            '<w:p><w:pPr><w:spacing w:before="0" w:after="60"/></w:pPr></w:p>')

# figure drawing paragraphs from the original (same rIds / aspect ratios; images are swapped in media/)
def _xml(el):
    return etree.tostring(el, encoding="unicode")
fig_xml = {}
for el in els:
    if el.find(".//w:drawing", ns) is not None:
        rid = el.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip").get(
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
        fig_xml[{"rId9": 1, "rId10": 2, "rId11": 3}[rid]] = _xml(el)

# Figures are one column wide and sit inline in the text flow, like the tables, so every viewer
# lays them out the same way and no full-width object can push text off a page.
import struct
COL_EMU = 4824 * 635                      # one column (4824 twips) in EMU

def fig(n, cap):
    run = re.search(r"<w:r>.*?<w:drawing>.*?</w:drawing></w:r>", fig_xml[n], re.S).group(0)
    with open(f"{D}/word/media/image{n}.png", "rb") as fh:
        w_px, h_px = struct.unpack(">II", fh.read(24)[16:24])
    cy = round(COL_EMU * h_px / w_px)
    run = re.sub(r'(<wp:extent|<a:ext) cx="\d+" cy="\d+"', lambda m: f'{m.group(1)} cx="{COL_EMU}" cy="{cy}"', run)
    run = run.replace("<wp:anchor", "<wp:inline")  # always inline
    return ('<w:p><w:pPr><w:keepNext/><w:spacing w:before="80" w:after="0"/><w:jc w:val="center"/></w:pPr>'
            f'{run}</w:p><w:p><w:pPr><w:spacing w:before="40" w:after="80"/><w:jc w:val="center"/></w:pPr>'
            + runs(cap, 16) + '</w:p>')

# ---------- assemble ----------
parts = []
for item in C.BODY:
    k = item[0]
    if k == "H1": parts.append(p_h1(item[1]))
    elif k == "H2": parts.append(p_h2(item[1]))
    elif k == "P": parts.append(p_body(item[1]))
    elif k == "EQ": parts.append(p_eq(item[1], item[2]))
    elif k == "TABLE":
        _, num, title, headers, rows, widths = item
        # tables sit inside one column (4824 twips) so they flow with the text
        parts += [caption_table(num, title), table(headers, rows, widths, 13)]
    elif k == "FIG":
        parts.append(fig(item[1], item[2]))
    elif k == "ALGO":
        parts.append(algo(item[1], item[2]))
    else:
        raise ValueError(k)

# the reference list starts on a new page that holds references only
parts.append(sect(2))   # continuous break: balances the two columns of the last text page
parts.append('<w:p><w:pPr><w:keepNext/><w:pageBreakBefore/><w:spacing w:before="0" w:after="40"/><w:jc w:val="center"/></w:pPr>'
             f'<w:r><w:rPr>{FONT}<w:b/><w:sz w:val="16"/></w:rPr><w:t>REFERENCES</w:t></w:r></w:p>')
for key, n in sorted(order.items(), key=lambda kv: kv[1]):
    parts.append('<w:p><w:pPr><w:spacing w:before="0" w:after="20" w:line="240" w:lineRule="auto"/>'
                 '<w:ind w:left="259" w:hanging="259"/><w:jc w:val="both"/></w:pPr>'
                 + runs(f"[{n}] " + C.REFS[key], 16) + '</w:p>')
parts.append(sect(2))

# title, abstract, keywords
title_p = ('<w:p><w:pPr><w:spacing w:after="160"/><w:jc w:val="center"/></w:pPr>'
           f'<w:r><w:rPr>{FONT}<w:sz w:val="48"/></w:rPr><w:t>{escape(C.TITLE)}</w:t></w:r></w:p>')
abstract = ('<w:p><w:pPr><w:spacing w:after="200"/><w:jc w:val="both"/></w:pPr>'
            f'<w:r><w:rPr>{FONT}<w:b/><w:i/><w:sz w:val="18"/></w:rPr><w:t>Abstract&#8212;</w:t></w:r>'
            + runs(C.ABSTRACT, 18, "<w:b/>") + '</w:p>')
keywords = ('<w:p><w:pPr><w:spacing w:after="140"/><w:jc w:val="both"/></w:pPr>'
            f'<w:r><w:rPr>{FONT}<w:b/><w:i/><w:sz w:val="18"/></w:rPr><w:t>Keywords&#8212;</w:t></w:r>'
            + runs(C.KEYWORDS, 18, "<w:i/>") + '</w:p>')

head = [title_p, _xml(els[1]), _xml(els[2]), _xml(els[3])]   # author table + first section break kept as-is
final_sect = _xml(els[-1])
new_body = "".join(head + [abstract, keywords] + parts) + final_sect

s = raw.decode("utf-8")
start = s.index("<w:body>") + len("<w:body>")
end = s.index("</w:body>")
# strip namespace declarations lxml adds when serializing subtrees
root_decl = set(re.findall(r'xmlns:(\w+)="', s[:s.index("<w:body>")]))
new_body = re.sub(r' xmlns:(\w+)="[^"]+"', lambda m: "" if m.group(1) in root_decl else m.group(0), new_body)
s = s[:start] + new_body + s[end:]
etree.fromstring(s.encode("utf-8"))  # well-formedness check
open(path, "w", encoding="utf-8").write(s)
print("refs:", len(order), "| parts:", len(parts))
