import codecs
import collections
import colorsys
import encodings
import functools
import itertools
import math
import operator
import os
import pathlib
import random
import re
import sys
from ast import literal_eval
from collections import defaultdict, OrderedDict, deque
from functools import reduce
from math import sqrt, floor, ceil
from urllib import parse
import numpy as np
import numexpr
import regex
from check_if_nan import is_nan
from divide_region_into_rectangles import divide_region_into_rect, cropimage
from flatten_any_dict_iterable_or_whatsoever import fla_tu
from flatten_everything import flatten_everything
from intersection_grouper import group_lists_with_intersections
from isiter import isiter
from locate_pixelcolor import search_colors
from rect_intersection import intersects
from tolerant_isinstance import isinstance_tolerant


@functools.lru_cache
def get_codecs():
    dir = encodings.__path__[0]
    codec_names = OrderedDict()
    for filename in os.listdir(dir):
        if not filename.endswith(".py"):
            continue
        name = filename[:-3]
        try:
            codec_names[name] = OrderedDict({"object": codecs.lookup(name)})
        except Exception as Fehler:
            pass
    return codec_names


class Trie:
    r"""
    Tr = Trie()
    Tr.trie_regex_from_words(['ich', 'du', 'er', 'sie', 'es', 'wir', 'der', 'die', 'das'])
    text = '.....'
    result = Tr.find(text)
    print(result)
    """

    def __init__(self):
        self.data = {}
        self.union = ""

    def add(self, word: str):
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            ref = ref[char]
        ref[""] = 1

    def dump(self):
        return self.data

    def quote(self, char):
        return re.escape(char)

    def _pattern(self, pData):
        data = pData
        if "" in data and len(data.keys()) == 1:
            return None

        alt = []
        cc = []
        q = 0
        for char in sorted(data.keys()):
            if isinstance(data[char], dict):
                try:
                    recurse = self._pattern(data[char])
                    alt.append(self.quote(char) + recurse)
                except Exception:
                    cc.append(self.quote(char))
            else:
                q = 1
        cconly = not len(alt) > 0

        if len(cc) > 0:
            if len(cc) == 1:
                alt.append(cc[0])
            else:
                alt.append("[" + "".join(cc) + "]")

        if len(alt) == 1:
            result = alt[0]
        else:
            result = "(?:" + "|".join(alt) + ")"

        if q:
            if cconly:
                result += "?"
            else:
                result = "(?:%s)?" % result
        return result

    def pattern(self):
        return self._pattern(self.dump())

    def trie_regex_from_words(
        self,
        words: list,
        boundary_right: bool = True,
        boundary_left: bool = True,
        capture: bool = False,
        ignorecase: bool = False,
        match_whole_line: bool = False,
    ):
        for word in words:
            self.add(word)
        anfang = ""
        ende = ""
        if match_whole_line is True:
            anfang += r"^\s*"
        if boundary_right is True:
            ende += r"\b"
        if capture is True:
            anfang += "("
        if boundary_left is True:
            anfang += r"\b"
        if capture is True:
            ende += ")"

        if match_whole_line is True:
            ende += r"\s*$"
        if ignorecase is True:
            self.union = regex.compile(anfang + self.pattern() + ende, regex.IGNORECASE)
        else:
            self.union = regex.compile(anfang + self.pattern() + ende)


def convert_to_normal_dict(di):
    if isinstance_tolerant(di, defaultdict):
        di = {k: convert_to_normal_dict(v) for k, v in di.items()}
    return di


def groupBy(key, seq, continue_on_exceptions=True, withindex=True, withvalue=True):
    indexcounter = -1

    def execute_f(k, v):
        nonlocal indexcounter
        indexcounter += 1
        try:
            return k(v)
        except Exception as fa:
            if continue_on_exceptions:
                return "EXCEPTION: " + str(fa)
            else:
                raise fa

    # based on https://stackoverflow.com/a/60282640/15096247
    if withvalue:
        return convert_to_normal_dict(
            reduce(
                lambda grp, val: grp[execute_f(key, val)].append(
                    val if not withindex else (indexcounter, val)
                )
                or grp,
                seq,
                defaultdict(list),
            )
        )
    return convert_to_normal_dict(
        reduce(
            lambda grp, val: grp[execute_f(key, val)].append(indexcounter) or grp,
            seq,
            defaultdict(list),
        )
    )


def iter_2_cycle_second_until_first_done(iter1, iter2):
    return ((x, y) for x, y in zip(iter1, itertools.cycle(iter2)))


def euclid_dist(x0, y0, x1, y1):
    dx_sq = (x0 - x1) ** 2
    dy_sq = (y0 - y1) ** 2
    return sqrt(dx_sq + dy_sq)


def groupby_euclid_dist(
    coord,
    seq,
    mindistance=0,
    maxdistance=500,
    continue_on_exceptions=True,
    withindex=False,
    withvalue=True,
):

    return groupBy(
        key=lambda x: True
        if (u := euclid_dist(*coord, *x)) >= mindistance and u <= maxdistance
        else False,
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_string_length(
    seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    return groupBy(
        key=lambda x: len(str(x)),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def group_values_in_flattened_nested_iter_and_count(
    seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    li = groupBy(
        key=lambda x: x,
        seq=flatten_everything(seq),
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )
    for key, item in li.copy().items():
        li[key] = len(item)
    return li


def groupby_type(seq, continue_on_exceptions=True, withindex=False, withvalue=True):
    return groupBy(
        key=lambda x: type(x),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_frequency(
    seq, continue_on_exceptions=True, withindex=False, withvalue=True
):

    return groupBy(
        key=lambda x: seq.count(x),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_can_be_divided_by(
    div, seq, continue_on_exceptions=True, withindex=False, withvalue=True
):

    return groupBy(
        key=lambda x: True if x % div == 0 else False,
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_division_remainder(
    div, seq, continue_on_exceptions=True, withindex=False, withvalue=True
):

    return groupBy(
        key=lambda x: divmod(x, div)[1],
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_divisor(
    div, seq, continue_on_exceptions=True, withindex=False, withvalue=True
):

    return groupBy(
        key=lambda x: divmod(x, div)[0],
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_bigger_than_or_equal(
    number, seq, continue_on_exceptions=True, withindex=False, withvalue=True
):

    return groupBy(
        key=lambda x: True if x >= number else False,
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_less_than_or_equal(
    number, seq, continue_on_exceptions=True, withindex=False, withvalue=True
):

    return groupBy(
        key=lambda x: True if x <= number else False,
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_bigger_than(
    number, seq, continue_on_exceptions=True, withindex=False, withvalue=True
):

    return groupBy(
        key=lambda x: True if x > number else False,
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_less_than(
    number, seq, continue_on_exceptions=True, withindex=False, withvalue=True
):

    return groupBy(
        key=lambda x: True if x < number else False,
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_equal(
    number, seq, continue_on_exceptions=True, withindex=False, withvalue=True
):

    return groupBy(
        key=lambda x: True if x == number else False,
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_regular_expression_matches(
    regexpressions, seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    if not isinstance_tolerant(regexpressions, list):
        regexpressions = [regexpressions]
    compr = [regex.compile(regexpression) for regexpression in regexpressions]

    def checkexp(compr, x):
        for co in compr:
            if co.search(str(x)) is not None:
                return True
        return False

    return groupBy(
        key=lambda x: checkexp(compr, x),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_is_integer(
    seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    return groupBy(
        key=lambda x: float(x).is_integer(),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_floor(seq, continue_on_exceptions=True, withindex=False, withvalue=True):
    return groupBy(
        key=lambda x: floor(x),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_ceil(seq, continue_on_exceptions=True, withindex=False, withvalue=True):
    return groupBy(
        key=lambda x: ceil(x),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_round(n, seq, continue_on_exceptions=True, withindex=False, withvalue=True):
    return groupBy(
        key=lambda x: round(x, n),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_endswith(
    n, seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    return groupBy(
        key=lambda x: str(x)[-n:],
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_startswith(
    n, seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    return groupBy(
        key=lambda x: str(x)[:n],
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_first_occurrence_in_string(
    char, seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    return groupBy(
        key=lambda x: str(x).find(char),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_isalnum(seq, continue_on_exceptions=True, withindex=False, withvalue=True):
    return groupBy(
        key=lambda x: x.isalnum(),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_isalpha(seq, continue_on_exceptions=True, withindex=False, withvalue=True):
    return groupBy(
        key=lambda x: x.isalpha(),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_isascii(seq, continue_on_exceptions=True, withindex=False, withvalue=True):
    return groupBy(
        key=lambda x: x.isascii(),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_isdecimal(
    seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    return groupBy(
        key=lambda x: x.isdecimal(),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_isdigit(seq, continue_on_exceptions=True, withindex=False, withvalue=True):
    return groupBy(
        key=lambda x: x.isdigit(),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_isidentifier(
    seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    return groupBy(
        key=lambda x: x.isidentifier(),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_islower(seq, continue_on_exceptions=True, withindex=False, withvalue=True):
    return groupBy(
        key=lambda x: x.islower(),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_isnumeric(
    seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    return groupBy(
        key=lambda x: x.isnumeric(),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_isprintable(
    seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    return groupBy(
        key=lambda x: x.isprintable(),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_isspace(seq, continue_on_exceptions=True, withindex=False, withvalue=True):
    return groupBy(
        key=lambda x: x.isspace(),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_istitle(seq, continue_on_exceptions=True, withindex=False, withvalue=True):
    return groupBy(
        key=lambda x: x.istitle(),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_isupper(seq, continue_on_exceptions=True, withindex=False, withvalue=True):
    return groupBy(
        key=lambda x: x.isupper(),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_last_occurrence_in_string(
    char, seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    return groupBy(
        key=lambda x: str(x).rfind(char),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_isin(
    value, seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    return groupBy(
        key=lambda x: value in x,
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_rectangle_intersects(
    rectangle, seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    return groupBy(
        key=lambda x: intersects(rectangle, x),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_isna(
    seq,
    emptyiters: bool = False,
    nastrings: bool = False,
    emptystrings: bool = False,
    emptybytes: bool = False,
    continue_on_exceptions=True,
    withindex=False,
    withvalue=True,
):
    return groupBy(
        key=lambda x: is_nan(
            x,
            emptyiters=emptyiters,
            nastrings=nastrings,
            emptystrings=emptystrings,
            emptybytes=emptybytes,
        ),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_isiter(seq, continue_on_exceptions=True, withindex=False, withvalue=True):
    return groupBy(
        key=lambda x: isiter(x),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_file_extension(
    seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    return groupBy(
        key=lambda x: str(pathlib.Path(x).suffix).lower(),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_color_qty_in_region(
    im,
    colors,
    startx=None,
    starty=None,
    stopx=None,
    stopy=None,
    rect_w=10,
    rect_h=10,
    continue_on_exceptions=True,
    withindex=False,
    withvalue=True,
):
    if startx is None:
        startx = 0
    if starty is None:
        starty = 0
    if stopx is None:
        stopx = im.shape[1]
    if stopy is None:
        stopy = im.shape[0]
    alre = divide_region_into_rect(
        startx=startx,
        starty=starty,
        w=stopx - startx,
        h=stopy - starty,
        square_w=rect_w,
        square_h=rect_h,
    )
    return groupBy(
        key=lambda x: (search_colors(cropimage(im, x), colors).shape[0]),
        seq=alre,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_even_odd(seq, continue_on_exceptions=True, withindex=False, withvalue=True):
    return groupBy(
        key=lambda x: "even" if x % 2 == 0 else "odd",
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_files_folder_link(
    seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    return groupBy(
        key=lambda x: "folder"
        if os.path.isdir(x)
        else "file"
        if os.path.isfile(x)
        else "link"
        if os.path.islink(x)
        else "unknown",
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_sys_size(seq, continue_on_exceptions=True, withindex=False, withvalue=True):
    return groupBy(
        key=lambda x: sys.getsizeof(x),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_first_item(
    seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    return groupBy(
        key=lambda x: x[0],
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_words_in_texts(
    textlist,
    wordlist,
    case_sen=False,
    continue_on_exceptions=True,
    withindex=False,
    boundary_right=True,
    boundary_left=True,
    withvalue=True,
):
    trie = Trie()
    trie.trie_regex_from_words(
        words=wordlist,
        boundary_right=boundary_right,
        boundary_left=boundary_left,
        capture=True,
        match_whole_line=False,
    )
    if not case_sen:
        compr = regex.compile(str(trie.union.pattern), flags=re.I)
    else:
        compr = regex.compile(str(trie.union.pattern))

    return groupBy(
        key=lambda x: tuple(
            [k for k in flatten_everything(compr.findall(x)) if k != ""]
        ),
        seq=textlist,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_sum(seq, continue_on_exceptions=True, withindex=False, withvalue=True):
    return groupBy(
        key=lambda x: sum(x),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def uri_validator(x):
    try:
        result = parse.urlparse(x)
        return all([result.scheme, result.netloc])
    except:
        return False


def groupby_valid_url(
    seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    return groupBy(
        key=lambda x: "valid" if uri_validator(x) else "not_valid",
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def iter_enumerate_multiple(*iterables):
    for it in zip(itertools.count(), *iterables):
        yield it


def iter__chain(*iterables):
    for it in iterables:
        for element in it:
            yield element


def iter_nested_dict(di):
    for x in fla_tu(di):
        keys = x[1]
        item = x[0]
        yield keys, item


def iter_split_list_into_chunks_fill_rest(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


def iter_cycle_shortest(iter1, iter2):
    turnaround = (len(iter2)) > (len(iter1))
    if turnaround:
        la = iter2
        lb = iter1
    else:
        la = iter1
        lb = iter2
    if turnaround:
        looplist = ((y, x) for x, y in zip(la, itertools.cycle(lb)))
    else:
        looplist = ((x, y) for x, y in zip(la, itertools.cycle(lb)))
    return looplist


def iter_reverse_lists_of_list(lis):
    for _ in (list(reversed(aa)) for aa in lis):
        yield _


def tee(iterable, n=2):
    it = iter(iterable)
    deques = [collections.deque() for i in range(n)]

    def gen(mydeque):
        while True:
            if not mydeque:  # when the local deque is empty
                try:
                    newval = next(it)  # fetch a new value and
                except StopIteration:
                    return
                for d in deques:  # load it to all the deques
                    d.append(newval)
            yield mydeque.popleft()

    return tuple(gen(d) for d in deques)


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def iter_split_by_index(iterable, indexes):
    return (iterable[p[0] : p[1]] for p in pairwise([0] + indexes + [len(iterable)]))


def iter_get_every_nth_element(iterable, start=0, step=2):
    for h in itertools.islice(iterable, start, len(iterable), step):
        yield h


def groupby_decoding_result(
    bytes_, mode="strict", continue_on_exceptions=True, withindex=False, withvalue=True
):
    seq = get_codecs()
    return groupBy(
        key=lambda x: bytes_.decode(x, mode),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_literal_eval_type(
    seq, continue_on_exceptions=True, withindex=False, withvalue=True
):
    return groupBy(
        key=lambda x: type((literal_eval(x))),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_percentage(
    seq,
    percent_true,
    group1=True,
    group2=False,
    continue_on_exceptions=True,
    withindex=False,
    withvalue=True,
):
    return groupBy(
        key=lambda x: random.choices(
            [group1, group2], [int(percent_true * 100), int(10000 - percent_true * 100)]
        )[0],
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_almost_equal(
    seq, value, equallimit, continue_on_exceptions=True, withindex=False, withvalue=True
):
    return groupBy(
        key=lambda x: abs(x - value) <= equallimit,
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def groupby_coords_almost_equal(
    seq,
    x_coord,
    y_coord,
    limit_x,
    limit_y,
    continue_on_exceptions=True,
    withindex=False,
    withvalue=True,
):
    return groupBy(
        key=lambda x: (abs(x[0] - x_coord) <= limit_x, abs(x[1] - y_coord) <= limit_y),
        seq=seq,
        continue_on_exceptions=continue_on_exceptions,
        withindex=withindex,
        withvalue=withvalue,
    )


def group_coordinates_by_distance(
    coordlist, limit_x, limit_y, continue_on_exceptions=True
):
    alltrues = []
    for _ in coordlist:
        xxx = groupby_coords_almost_equal(
            seq=coordlist,
            x_coord=_[0],
            y_coord=_[1],
            limit_x=limit_x,
            limit_y=limit_y,
            continue_on_exceptions=continue_on_exceptions,
            withindex=False,
            withvalue=True,
        )
        alltrues.append(xxx.get((True, True)))
    return group_lists_with_intersections(alltrues, keep_duplicates=False)


def iter_find_same_ending_elements(iters):
    iterables = [list(reversed(x)) for x in iters]
    return reversed(
        [x[0] for x in itertools.takewhile(lambda x: len(set(x)) == 1, zip(*iterables))]
    )


def iter_find_same_beginning_elements(iters):
    return (x[0] for x in itertools.takewhile(lambda x: len(set(x)) == 1, zip(*iters)))


def iter_nested_list_from_item(variable, shape):
    return reduce(
        lambda a, b: list(itertools.repeat(a, b)),
        shape,
        variable,
    )


def iter_repeat_items(iterable, reps):
    return (x for x in iterable for i in range(reps))


def iter_count_occurrences_of_item_in_multiple_iterables(iterables):
    return collections.Counter(itertools.chain.from_iterable(iterables))


def iter_count_occurrences_of_item_in_multiple_iterables_flatten_everything(iterables):
    return collections.Counter(flatten_everything(iterables))


def iter_values_from_list1_based_on_values_of_list2(list1, list2, condition):
    return itertools.compress(list1, map(condition, list2))


def iter_windowed(iterable, n):
    accum = deque((), n)
    for element in iterable:
        accum.append(element)
        if len(accum) == n:
            yield tuple(accum)


def iter_batch(iterable, n):
    # https://stackoverflow.com/a/74997058/15096247
    _consume = collections.deque(maxlen=0).extend
    "Batch data into sub-iterators of length n. The last batch may be shorter."
    # batched_it('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    n -= (
        1  # First element pulled for us, pre-decrement n so we don't redo it every loop
    )
    it = iter(iterable)
    for first_el in it:
        chunk_it = itertools.islice(it, n)
        try:
            yield itertools.chain((first_el,), chunk_it)
        finally:
            _consume(chunk_it)  # Efficiently consume any elements caller didn't consume


def numpy_all_permutations(iterable, len_of_each):
    def permutations_(n, k):
        if n <= 256:
            dtype = np.uint8
        else:
            dtype = np.uint16
        a = np.zeros((math.perm(n, k), k), dtype)
        f = 1
        for m in range(n - k + 1, n + 1):
            b = a[:f, n - m + 1 :]
            for i in range(1, m):
                a[i * f : (i + 1) * f, n - m] = i
                a[i * f : (i + 1) * f, n - m + 1 :] = b + (b >= i)
            b += 1
            f *= m
        return a

    alpha = iterable
    alpha1 = [x for x in enumerate(alpha)]
    arra = permutations_(len(alpha1), len_of_each)
    condlist = []
    for _ in alpha1:
        condlist.append(numexpr.evaluate(f"arra=={_[0]}"))
    if not isinstance_tolerant(alpha, np.ndarray):
        choicelist = np.array(alpha)
    else:
        choicelist = alpha
    stan_val = choicelist[0]
    return np.select(condlist, choicelist, stan_val)


def numpy_random_array_no_reps_beginning_end(iterable, block_count):
    def generate_random_array(block_length, block_count):
        # https://stackoverflow.com/a/52556609/15096247
        for blocks in range(0, block_count):
            nums = np.arange(block_length)
            np.random.shuffle(nums)
            try:
                if nums[0] == randoms_array[-1]:
                    nums[0], nums[-1] = nums[-1], nums[0]
            except NameError:
                randoms_array = []
            randoms_array.extend(nums)
        return randoms_array

    alpha = iterable
    alpha1 = [x for x in enumerate(alpha)]
    arra = generate_random_array(len(alpha1), block_count)
    condlist = []
    for _ in alpha1:
        condlist.append(numexpr.evaluate(f"arra=={_[0]}"))
    if not isinstance_tolerant(alpha, np.ndarray):
        choicelist = np.array(alpha)
    else:
        choicelist = alpha
    stan_val = choicelist[0]
    return np.select(condlist, choicelist, stan_val).reshape((block_count, -1))


def iter_nested_for_loop(*args):
    for vals in itertools.product(*args):
        yield vals


def iter_nested_for_loop_enumerate(*args):
    co = 0
    for vals in itertools.product(*args):
        yield (co,) + vals
        co += 1


def iter_chain_flatten(*args):
    for _ in flatten_everything(args):
        yield _


def iter_spiral(radius, eccentricity=1.5, step=0.1, t=0):
    while True:
        t += step
        yield eccentricity * radius * t * math.cos(t), radius * t * math.sin(t)


def iter_add_one_item_each_iteration(iterable):
    for i in range(len(iterable)):
        yield iterable[: i + 1]


def iter_add_one_item_each_iteration_reverse(iterable):
    l = len(iterable)
    for i in range(l):
        yield iterable[l - i - 1 :]


def iter_nested(iterable):
    try:
        vara = iterable.copy()
    except Exception:
        vara = iterable
    for x in fla_tu(vara):
        n = len(x[1])
        foryield = []
        for i in range(n):
            foryield.append(reduce(operator.getitem, x[1][: i + 1], vara))
        yield foryield


def iter_nested_with_path(iterable):
    try:
        vara = iterable.copy()
    except Exception:
        vara = iterable
    for x in fla_tu(vara):
        n = len(x[1])
        foryield = []
        for i in range(n):
            foryield.append(
                (x[1][: i + 1], reduce(operator.getitem, x[1][: i + 1], vara))
            )
        yield foryield


def iter_rainbow_colors(num_colors):
    colors = []
    for i in np.arange(0.0, 360.0, 360.0 / num_colors):
        hue = i / 360.0
        lightness = (50 + np.random.rand() * 10) / 100.0
        saturation = (90 + np.random.rand() * 10) / 100.0
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    for _ in ((int(255 * o[0]), int(255 * o[1]), int(255 * o[2])) for o in colors):
        yield _


def iter_reshape(seq, how):
    """
    # based on sympy.utilities.reshape
    """
    m = sum(flatten_everything(how))
    seq = list(flatten_everything(seq))
    n, rem = divmod(len(seq), m)
    if m < 0 or rem:
        raise ValueError(
            "template must sum to positive number "
            "that divides the length of the sequence"
        )
    i = 0
    container = type(how)
    rv = [None] * n
    for k in range(len(rv)):
        _rv = []
        for hi in how:
            if isinstance(hi, int):
                _rv.extend(seq[i : i + hi])
                i += hi
            else:
                n = sum(list(flatten_everything(hi)))
                hi_type = type(hi)
                _rv.append(hi_type(iter_reshape(seq[i : i + n], hi)[0]))
                i += n
        rv[k] = container(_rv)
    return type(seq)(rv)


def iter_rotate_left(iterable, n, onlyfinal=False):
    try:
        iterable_ = iterable.copy()
    except Exception:
        iterable_ = iterable

    for _ in range(n):
        iterable_ = iterable_[1:] + iterable_[:1]
        if not onlyfinal:
            yield iterable_
    if onlyfinal:
        yield iterable_


def iter_rotate_right(iterable, n, onlyfinal=False):
    try:
        iterable_ = iterable.copy()
    except Exception:
        iterable_ = iterable

    for _ in range(n):
        iterable_ = iterable_[-1:] + iterable_[:-1]
        if not onlyfinal:
            yield iterable_
    if onlyfinal:
        yield iterable_


def polymul(p, q):
    """
    Multiply two polynomials, represented as lists of coefficients.
    """
    r = [0] * (len(p) + len(q) - 1)
    for i, c in enumerate(p):
        for j, d in enumerate(q):
            r[i + j] += c * d
    return r


def iter_number_of_combinations(it, k):
    """
    from https://stackoverflow.com/a/48612518/15096247
    Number of combinations of length *k* of the elements of *it*.
    """
    counts = collections.Counter(it).values()
    prod = reduce(polymul, [[1] * (count + 1) for count in counts], [1])
    return prod[k] if k < len(prod) else 0
