# Many useful groupby / itertools functions 

```python
pip install group-and-iter-everything
```

### groupby stuff 

```python


######################################################
# Almost all groupby functions accept the kwargs: continue_on_exceptions, withindex, withvalue


# continue_on_exceptions=True - Exceptions are pulled, and each Exception gets its own group
print(groupby_can_be_divided_by(
    div=2, seq=(["1", 1, 3, 3, 3, 5, 3, 5, 22.22, *(3, 4), *[333, 4]]), continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {'EXCEPTION: not all arguments converted during string formatting': ['1'],
# False: [1, 3, 3, 3, 5, 3, 5, 22.22, 3, 333],
# True: [4, 4]}
 
######################################################
# withindex=True - Returns the index and the value of the input list:
print(groupby_can_be_divided_by(
    div=2, seq=(["1", 1, 3, 3, 3, 5, 3, 5, 22.22, *(3, 4), *[333, 4]]), continue_on_exceptions=True, withindex=True, withvalue=True
))

# output:{'EXCEPTION: not all arguments converted during string formatting': [(0, '1')], 
#  False: [(1, 1), (2, 3), (3, 3), (4, 3), (5, 5), (6, 3), (7, 5), (8, 22.22), (9, 3), (11, 333)], 
#  True: [(10, 4), (12, 4)]}


######################################################
# withvalue=False - Only the index of the input list is returned:
# output: {'EXCEPTION: not all arguments converted during string formatting': [0], 
# False: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11],
# True: [10, 12]}
######################################################

```



```python
######################################################
print(list(iter_2_cycle_second_until_first_done(iter1=[3, 3, 4, 4], iter2=[1, 2])))
# output: [(3, 1), (3, 2), (4, 1), (4, 2)]
######################################################
print(groupby_euclid_dist(
    coord=(50, 100),
    seq=[(100, 100), (300, 500), (900, 23), (10, 10)],
    mindistance=0,
    maxdistance=500,
))

# output: {True: [(100, 100), (300, 500), (10, 10)], False: [(900, 23)]}
######################################################
print(groupby_string_length(
    seq=["dd", "b", "ff", "saaa"],
))

# output: {2: ['dd', 'ff'], 1: ['b'], 4: ['saaa']}
######################################################
print(group_values_in_flattened_nested_iter_and_count(seq=[1,2,3,4,[33,412,2,2,3], {(3,4), (1,4)}], continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {1: 2, 2: 3, 3: 3, 4: 3, 33: 1, 412: 1}
######################################################

print(groupby_type(seq=["dd", "b", "ff", "saaa", 1, 3, 5, 22.22, (3, 4), [333, 4]]))
# output: {<class 'str'>: ['dd', 'b', 'ff', 'saaa'], <class 'int'>: [1, 3, 5], <class 'float'>: [22.22], <class 'tuple'>: [(3, 4)], <class 'list'>: [[333, 4]]}
######################################################
print(groupby_frequency(
    seq=[
        "dd",
        "b",
        "ff",
        "saaa",
        1,
        3,
        3,
        3,
        5,
        "b",
        3,
        5,
        22.22,
        (3, 4),
        [333, 4],
    ]
))
# output: {1: ['dd', 'ff', 'saaa', 1, 22.22, (3, 4), [333, 4]], 2: ['b', 5, 'b', 5], 4: [3, 3, 3, 3]}
######################################################
print(groupby_division_remainder(
    div=2, seq=([1, 3, 3, 3, 5, 3, 5, 22.22, *(3, 4), *[333, 4]])
))
# output: {1: [1, 3, 3, 3, 5, 3, 5, 3, 333], 0.21999999999999886: [22.22], 0: [4, 4]}
######################################################
print(groupby_divisor(
    div=2, seq=([1, 3, 3, 3, 5, 3, 5, 22.22, *(3, 4), *[333, 4]])
))
# output: {0: [1], 1: [3, 3, 3, 3, 3], 2: [5, 5, 4, 4], 11.0: [22.22], 166: [333]}
######################################################
print(groupby_less_than_or_equal(
    number=2,
    seq=([1, 3, 3, 3, 5, 3,2,2, 5, 22.22, *(3, 4), *[333, 4]]),
    continue_on_exceptions=True,
    withindex=False,
))
# output: {True: [1, 2, 2], False: [3, 3, 3, 5, 3, 5, 22.22, 3, 4, 333, 4]}
######################################################
print(groupby_less_than(
    number=2,
    seq=([1, 3, 3, 3, 5, 3, 5, 22.22, *(3, 4), *[333, 4]]),
    continue_on_exceptions=True,
    withindex=False,
))
# output: {True: [1], False: [3, 3, 3, 5, 3, 5, 22.22, 3, 4, 333, 4]}
######################################################
print(groupby_bigger_than_or_equal(
    number=2,
    seq=([1, 3, 3, 3, 5, 3, 2,2,2,5, 22.22, *(3, 4), *[333, 4]]),
    continue_on_exceptions=True,
    withindex=False,
))
# output: {False: [1], True: [3, 3, 3, 5, 3, 2, 2, 2, 5, 22.22, 3, 4, 333, 4]}
######################################################
print(groupby_bigger_than(
    number=2,
    seq=(["1", 1, 3, 3, 3, 5, 3, 5, 22.22, *(3, 4), *[333, 4]]),
    continue_on_exceptions=True,
    withindex=False,
))
# output: {"EXCEPTION: '>' not supported between instances of 'str' and 'int'": ['1'], False: [1], True: [3, 3, 3, 5, 3, 5, 22.22, 3, 4, 333, 4]}
######################################################
print(groupby_equal(
    number=3,
    seq=(["1", 1, 3, 3, 3, 5, 3, 5, 22.22, *(3, 4), *[333, 4]]),
    continue_on_exceptions=True,
    withindex=False,
))
# output: {False: ['1', 1, 5, 5, 22.22, 4, 333, 4], True: [3, 3, 3, 3, 3]}
######################################################
print(groupby_regular_expression_matches(
    regexpressions=[r"^\d+$", '^A+$'],
    seq=([1, 3, 3, 3, 5, 3, 5,'AA', 'BB', 22.22, *(3, 4), *[333, 4]]),
    continue_on_exceptions=True,
    withindex=False,
))

# output: {True: [1, 3, 3, 3, 5, 3, 5, 'AA', 3, 4, 333, 4], False: ['BB', 22.22]}
######################################################
print(groupby_is_integer(
    seq=[2.0,1,1,3,5,3.0, 3.4,2.4,25.3], continue_on_exceptions=True, withindex=False, withvalue=True))
# output: {True: [2.0, 1, 1, 3, 5, 3.0], False: [3.4, 2.4, 25.3]}
######################################################
print(groupby_floor(
    seq=[1.0,1.1,1.2,1.3,1.9,1.8,2.0,1,1,3,5,3.0, 3.4,2.4,25.3], continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {1: [1.0, 1.1, 1.2, 1.3, 1.9, 1.8, 1, 1], 2: [2.0, 2.4], 3: [3, 3.0, 3.4], 5: [5], 25: [25.3]}
######################################################
print(groupby_ceil(
    seq=[1.0,1.1,1.2,1.3,1.9,1.8,2.0,1,1,3,5,3.0, 3.4,2.4,25.3], continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {1: [1.0, 1, 1], 2: [1.1, 1.2, 1.3, 1.9, 1.8, 2.0], 3: [3, 3.0, 2.4], 5: [5], 4: [3.4], 26: [25.3]}
######################################################
print(groupby_round(n=2,
    seq=[1.11111111,1.111222222,1.11113334,1.3,1.9,1.8,2.0,1,1,3,5,3.0, 3.4,2.4,25.3], continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {1.11: [1.11111111, 1.111222222, 1.11113334], 1.3: [1.3], 1.9: [1.9], 1.8: [1.8], 2.0: [2.0], 1: [1, 1], 3: [3, 3.0], 5: [5], 3.4: [3.4], 2.4: [2.4], 25.3: [25.3]}
######################################################
print(groupby_endswith(
    n=1,seq=['hallo', 'boot', 'baba', 'bdoo', 'flaot', 'mama' ], continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {'o': ['hallo', 'bdoo'], 't': ['boot', 'flaot'], 'a': ['baba', 'mama']}
######################################################
print(groupby_startswith(
    n=1,seq=['hallo', 'boot', 'baba', 'bdoo', 'flaot', 'hmama' ], continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {'h': ['hallo', 'hmama'], 'b': ['boot', 'baba', 'bdoo'], 'f': ['flaot']}

print(groupby_startswith(
    n=1,seq=['hallo', 'boot', 'baba', 'bdoo', 'flaot', 'hmama' ], continue_on_exceptions=True, withindex=True, withvalue=False
))
# output: {'h': [0, 5], 'b': [1, 2, 3], 'f': [4]}

print(groupby_startswith(
    n=1,seq=['hallo', 'boot', 'baba', 'bdoo', 'flaot', 'hmama' ], continue_on_exceptions=True, withindex=True, withvalue=True
))
# output: {'h': [(0, 'hallo'), (5, 'hmama')], 'b': [(1, 'boot'), (2, 'baba'), (3, 'bdoo')], 'f': [(4, 'flaot')]}
######################################################
print(groupby_first_occurrence_in_string(
    char='a',seq=['hallo', 'boot', 'baba', 'bdoo', 'flaot', 'hmama' ], continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {1: ['hallo', 'baba'], -1: ['boot', 'bdoo'], 2: ['flaot', 'hmama']}
######################################################
print(groupby_last_occurrence_in_string(
    char='a',seq=['hallo', 'boot', 'baba', 'bdoo', 'flaot', 'hmama' ], continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {1: ['hallo'], -1: ['boot', 'bdoo'], 3: ['baba'], 2: ['flaot'], 4: ['hmama']}
######################################################
text = '''One evening, a few days later, K. was walking along one of the corridors that separated his office from the main stairway—he was nearly the last one to leave for home that evening, there remained only a couple of workers in the light of a single bulb in the dispatch department—when he heard a sigh from behind a door which he had himself never opened but which he had always thought just led into a junk room. He stood in amazement and listened again to establish whether he might not be mistaken'''.split()
print(groupby_isalnum(
    seq=text, continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {True: ['One', 'a', 'few', 'days', 'was', 'walking', 'along', 'one', 'of', 'the', 'corridors', 'that', 'separated', 'his', 'office', 'from', 'the', 'main', 'was', 'nearly', 'the', 'last', 'one', 'to', 'leave', 'for', 'home', 'that', 'there', 'remained', 'only', 'a', 'couple', 'of', 'workers', 'in', 'the', 'light', 'of', 'a', 'single', 'bulb', 'in', 'the', 'dispatch', 'he', 'heard', 'a', 'sigh', 'from', 'behind', 'a', 'door', 'which', 'he', 'had', 'himself', 'never', 'opened', 'but', 'which', 'he', 'had', 'always', 'thought', 'just', 'led', 'into', 'a', 'junk', 'He', 'stood', 'in', 'amazement', 'and', 'listened', 'again', 'to', 'establish', 'whether', 'he', 'might', 'not', 'be', 'mistaken'], False: ['evening,', 'later,', 'K.', 'stairway—he', 'evening,', 'department—when', 'room.']}
######################################################
print(groupby_isalpha(
    seq=text, continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {True: ['One', 'a', 'few', 'days', 'was', 'walking', 'along', 'one', 'of', 'the', 'corridors', 'that', 'separated', 'his', 'office', 'from', 'the', 'main', 'was', 'nearly', 'the', 'last', 'one', 'to', 'leave', 'for', 'home', 'that', 'there', 'remained', 'only', 'a', 'couple', 'of', 'workers', 'in', 'the', 'light', 'of', 'a', 'single', 'bulb', 'in', 'the', 'dispatch', 'he', 'heard', 'a', 'sigh', 'from', 'behind', 'a', 'door', 'which', 'he', 'had', 'himself', 'never', 'opened', 'but', 'which', 'he', 'had', 'always', 'thought', 'just', 'led', 'into', 'a', 'junk', 'He', 'stood', 'in', 'amazement', 'and', 'listened', 'again', 'to', 'establish', 'whether', 'he', 'might', 'not', 'be', 'mistaken'], False: ['evening,', 'later,', 'K.', 'stairway—he', 'evening,', 'department—when', 'room.']}
######################################################
print(groupby_isascii(
    seq=text, continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {True: ['One', 'evening,', 'a', 'few', 'days', 'later,', 'K.', 'was', 'walking', 'along', 'one', 'of', 'the', 'corridors', 'that', 'separated', 'his', 'office', 'from', 'the', 'main', 'was', 'nearly', 'the', 'last', 'one', 'to', 'leave', 'for', 'home', 'that', 'evening,', 'there', 'remained', 'only', 'a', 'couple', 'of', 'workers', 'in', 'the', 'light', 'of', 'a', 'single', 'bulb', 'in', 'the', 'dispatch', 'he', 'heard', 'a', 'sigh', 'from', 'behind', 'a', 'door', 'which', 'he', 'had', 'himself', 'never', 'opened', 'but', 'which', 'he', 'had', 'always', 'thought', 'just', 'led', 'into', 'a', 'junk', 'room.', 'He', 'stood', 'in', 'amazement', 'and', 'listened', 'again', 'to', 'establish', 'whether', 'he', 'might', 'not', 'be', 'mistaken'], False: ['stairway—he', 'department—when']}
######################################################
print(groupby_isdecimal(
    seq=text, continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {False: ['One', 'evening,', 'a', 'few', 'days', 'later,', 'K.', 'was', 'walking', 'along', 'one', 'of', 'the', 'corridors', 'that', 'separated', 'his', 'office', 'from', 'the', 'main', 'stairway—he', 'was', 'nearly', 'the', 'last', 'one', 'to', 'leave', 'for', 'home', 'that', 'evening,', 'there', 'remained', 'only', 'a', 'couple', 'of', 'workers', 'in', 'the', 'light', 'of', 'a', 'single', 'bulb', 'in', 'the', 'dispatch', 'department—when', 'he', 'heard', 'a', 'sigh', 'from', 'behind', 'a', 'door', 'which', 'he', 'had', 'himself', 'never', 'opened', 'but', 'which', 'he', 'had', 'always', 'thought', 'just', 'led', 'into', 'a', 'junk', 'room.', 'He', 'stood', 'in', 'amazement', 'and', 'listened', 'again', 'to', 'establish', 'whether', 'he', 'might', 'not', 'be', 'mistaken']}
######################################################
print(groupby_isdigit(
    seq=text, continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {False: ['One', 'evening,', 'a', 'few', 'days', 'later,', 'K.', 'was', 'walking', 'along', 'one', 'of', 'the', 'corridors', 'that', 'separated', 'his', 'office', 'from', 'the', 'main', 'stairway—he', 'was', 'nearly', 'the', 'last', 'one', 'to', 'leave', 'for', 'home', 'that', 'evening,', 'there', 'remained', 'only', 'a', 'couple', 'of', 'workers', 'in', 'the', 'light', 'of', 'a', 'single', 'bulb', 'in', 'the', 'dispatch', 'department—when', 'he', 'heard', 'a', 'sigh', 'from', 'behind', 'a', 'door', 'which', 'he', 'had', 'himself', 'never', 'opened', 'but', 'which', 'he', 'had', 'always', 'thought', 'just', 'led', 'into', 'a', 'junk', 'room.', 'He', 'stood', 'in', 'amazement', 'and', 'listened', 'again', 'to', 'establish', 'whether', 'he', 'might', 'not', 'be', 'mistaken']}
######################################################
print(groupby_isidentifier(
    seq=text, continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {True: ['One', 'a', 'few', 'days', 'was', 'walking', 'along', 'one', 'of', 'the', 'corridors', 'that', 'separated', 'his', 'office', 'from', 'the', 'main', 'was', 'nearly', 'the', 'last', 'one', 'to', 'leave', 'for', 'home', 'that', 'there', 'remained', 'only', 'a', 'couple', 'of', 'workers', 'in', 'the', 'light', 'of', 'a', 'single', 'bulb', 'in', 'the', 'dispatch', 'he', 'heard', 'a', 'sigh', 'from', 'behind', 'a', 'door', 'which', 'he', 'had', 'himself', 'never', 'opened', 'but', 'which', 'he', 'had', 'always', 'thought', 'just', 'led', 'into', 'a', 'junk', 'He', 'stood', 'in', 'amazement', 'and', 'listened', 'again', 'to', 'establish', 'whether', 'he', 'might', 'not', 'be', 'mistaken'], False: ['evening,', 'later,', 'K.', 'stairway—he', 'evening,', 'department—when', 'room.']}
######################################################
print(groupby_islower(
    seq=text, continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {False: ['One', 'K.', 'He'], True: ['evening,', 'a', 'few', 'days', 'later,', 'was', 'walking', 'along', 'one', 'of', 'the', 'corridors', 'that', 'separated', 'his', 'office', 'from', 'the', 'main', 'stairway—he', 'was', 'nearly', 'the', 'last', 'one', 'to', 'leave', 'for', 'home', 'that', 'evening,', 'there', 'remained', 'only', 'a', 'couple', 'of', 'workers', 'in', 'the', 'light', 'of', 'a', 'single', 'bulb', 'in', 'the', 'dispatch', 'department—when', 'he', 'heard', 'a', 'sigh', 'from', 'behind', 'a', 'door', 'which', 'he', 'had', 'himself', 'never', 'opened', 'but', 'which', 'he', 'had', 'always', 'thought', 'just', 'led', 'into', 'a', 'junk', 'room.', 'stood', 'in', 'amazement', 'and', 'listened', 'again', 'to', 'establish', 'whether', 'he', 'might', 'not', 'be', 'mistaken']}
######################################################
print(groupby_isnumeric(
    seq=text, continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {False: ['One', 'evening,', 'a', 'few', 'days', 'later,', 'K.', 'was', 'walking', 'along', 'one', 'of', 'the', 'corridors', 'that', 'separated', 'his', 'office', 'from', 'the', 'main', 'stairway—he', 'was', 'nearly', 'the', 'last', 'one', 'to', 'leave', 'for', 'home', 'that', 'evening,', 'there', 'remained', 'only', 'a', 'couple', 'of', 'workers', 'in', 'the', 'light', 'of', 'a', 'single', 'bulb', 'in', 'the', 'dispatch', 'department—when', 'he', 'heard', 'a', 'sigh', 'from', 'behind', 'a', 'door', 'which', 'he', 'had', 'himself', 'never', 'opened', 'but', 'which', 'he', 'had', 'always', 'thought', 'just', 'led', 'into', 'a', 'junk', 'room.', 'He', 'stood', 'in', 'amazement', 'and', 'listened', 'again', 'to', 'establish', 'whether', 'he', 'might', 'not', 'be', 'mistaken']}
######################################################
print(groupby_isprintable(
    seq=text, continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {True: ['One', 'evening,', 'a', 'few', 'days', 'later,', 'K.', 'was', 'walking', 'along', 'one', 'of', 'the', 'corridors', 'that', 'separated', 'his', 'office', 'from', 'the', 'main', 'stairway—he', 'was', 'nearly', 'the', 'last', 'one', 'to', 'leave', 'for', 'home', 'that', 'evening,', 'there', 'remained', 'only', 'a', 'couple', 'of', 'workers', 'in', 'the', 'light', 'of', 'a', 'single', 'bulb', 'in', 'the', 'dispatch', 'department—when', 'he', 'heard', 'a', 'sigh', 'from', 'behind', 'a', 'door', 'which', 'he', 'had', 'himself', 'never', 'opened', 'but', 'which', 'he', 'had', 'always', 'thought', 'just', 'led', 'into', 'a', 'junk', 'room.', 'He', 'stood', 'in', 'amazement', 'and', 'listened', 'again', 'to', 'establish', 'whether', 'he', 'might', 'not', 'be', 'mistaken']}
######################################################
print(groupby_isspace(
    seq=text, continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {False: ['One', 'evening,', 'a', 'few', 'days', 'later,', 'K.', 'was', 'walking', 'along', 'one', 'of', 'the', 'corridors', 'that', 'separated', 'his', 'office', 'from', 'the', 'main', 'stairway—he', 'was', 'nearly', 'the', 'last', 'one', 'to', 'leave', 'for', 'home', 'that', 'evening,', 'there', 'remained', 'only', 'a', 'couple', 'of', 'workers', 'in', 'the', 'light', 'of', 'a', 'single', 'bulb', 'in', 'the', 'dispatch', 'department—when', 'he', 'heard', 'a', 'sigh', 'from', 'behind', 'a', 'door', 'which', 'he', 'had', 'himself', 'never', 'opened', 'but', 'which', 'he', 'had', 'always', 'thought', 'just', 'led', 'into', 'a', 'junk', 'room.', 'He', 'stood', 'in', 'amazement', 'and', 'listened', 'again', 'to', 'establish', 'whether', 'he', 'might', 'not', 'be', 'mistaken']}
######################################################
print(groupby_istitle(
    seq=text, continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {True: ['One', 'K.', 'He'], False: ['evening,', 'a', 'few', 'days', 'later,', 'was', 'walking', 'along', 'one', 'of', 'the', 'corridors', 'that', 'separated', 'his', 'office', 'from', 'the', 'main', 'stairway—he', 'was', 'nearly', 'the', 'last', 'one', 'to', 'leave', 'for', 'home', 'that', 'evening,', 'there', 'remained', 'only', 'a', 'couple', 'of', 'workers', 'in', 'the', 'light', 'of', 'a', 'single', 'bulb', 'in', 'the', 'dispatch', 'department—when', 'he', 'heard', 'a', 'sigh', 'from', 'behind', 'a', 'door', 'which', 'he', 'had', 'himself', 'never', 'opened', 'but', 'which', 'he', 'had', 'always', 'thought', 'just', 'led', 'into', 'a', 'junk', 'room.', 'stood', 'in', 'amazement', 'and', 'listened', 'again', 'to', 'establish', 'whether', 'he', 'might', 'not', 'be', 'mistaken']}
######################################################
print(groupby_isupper(
    seq=text, continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {False: ['One', 'evening,', 'a', 'few', 'days', 'later,', 'was', 'walking', 'along', 'one', 'of', 'the', 'corridors', 'that', 'separated', 'his', 'office', 'from', 'the', 'main', 'stairway—he', 'was', 'nearly', 'the', 'last', 'one', 'to', 'leave', 'for', 'home', 'that', 'evening,', 'there', 'remained', 'only', 'a', 'couple', 'of', 'workers', 'in', 'the', 'light', 'of', 'a', 'single', 'bulb', 'in', 'the', 'dispatch', 'department—when', 'he', 'heard', 'a', 'sigh', 'from', 'behind', 'a', 'door', 'which', 'he', 'had', 'himself', 'never', 'opened', 'but', 'which', 'he', 'had', 'always', 'thought', 'just', 'led', 'into', 'a', 'junk', 'room.', 'He', 'stood', 'in', 'amazement', 'and', 'listened', 'again', 'to', 'establish', 'whether', 'he', 'might', 'not', 'be', 'mistaken'], True: ['K.']}
######################################################
print(groupby_isin(value='1', seq=['1', '11', '2', 1,2,3,54,2], continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {True: ['1', '11'], False: ['2'], "EXCEPTION: argument of type 'int' is not iterable": [1, 2, 3, 54, 2]}
######################################################
import random
print(groupby_rectangle_intersects(
    rectangle=(0, 0, 100, 100),
    seq=[
        (
            g := random.randrange(1, 200),
            f := random.randrange(1, 200),
            g + 100,
            f + 100,
        )
        for _ in range(10)
    ],
    continue_on_exceptions=True,
    withindex=True,
))
# output: {False: [(0, (88, 157, 188, 257)), (1, (176, 63, 276, 163)), (2, (38, 102, 138, 202)), (4, (159, 145, 259, 245)), (5, (84, 199, 184, 299)), (6, (6, 160, 106, 260)), (7, (101, 133, 201, 233)), (9, (188, 148, 288, 248))], True: [(3, (27, 68, 127, 168)), (8, (2, 58, 102, 158))]}
######################################################
import pandas as pd
import math
import numpy as np
print(groupby_isna(
    seq=[pd.NA, np.nan, None, math.nan, 3,54,3,(22,34,412), {323,31}, [3312,3]],
    emptyiters = False,
    nastrings = False,
    emptystrings = False,
    emptybytes = False,
    continue_on_exceptions=True,
    withindex=False,
    withvalue=True,
))
# output: {True: [<NA>, nan, None, nan], False: [3, 54, 3, (22, 34, 412), {323, 31}, [3312, 3]]}
######################################################
import pandas as pd
import math
import numpy as np
print(groupby_isiter(
    seq=[pd.NA, np.nan, None, math.nan, 3, 54, 3, (22, 34, 412), {323, 31}, [3312, 3]],
    continue_on_exceptions=True,
    withindex=False,
    withvalue=True,
))
# output: {False: [<NA>, nan, None, nan, 3, 54, 3], True: [(22, 34, 412), {323, 31}, [3312, 3]]}
######################################################
import os
print(groupby_file_extension(
    seq=os.listdir(r'F:\gitrep\screenshots'), continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {'': ['.git', '.gitignore', 'color', 'debugdf', 'enumeratefiles', 'LICENSE', 'sortfiles'], '.png': ['144.png', '146.png', '2022-12-27 03_40_27-.png', '2022-12-27 03_49_27-.png', '2022-12-27 04_01_57-.png', '2022-12-27 04_02_09-.png', '2023-01-01 09_48_27-single_elements.png', 'add_methods.png', 'allcolors.png', 'asciiart.png', 'bluestacksauto.png', 'buttonsearch.png', 'collage.png', 'colorfind1.png', 'colorfind1xxx.png', 'colorfind2.png', 'colorfind3.png', 'colorfind4.png', 'cv2_putTrueTypeText_000000.png', 'cv2_putTrueTypeText_000001.png', 'cv2_putTrueTypeText_000002.png', 'cv2_putTrueTypeText_000003.png', 'cv2_putTrueTypeText_000004.png', 'cv2_putTrueTypeText_000005.png', 'cv2_putTrueTypeText_000006.png', 'cv2_putTrueTypeText_000007.png', 'cv2_putTrueTypeText_000008.png', 'cv2_putTrueTypeText_000009.png', 'cv2_putTrueTypeText_000010.png', 'cv2_putTrueTypeText_000011.png', 'cv2_putTrueTypeText_000012.png', 'cv2_putTrueTypeText_000013.png', 'cv2_putTrueTypeText_000014.png', 'cv2_putTrueTypeText_000015.png', 'cv2_putTrueTypeText_000017.png', 'dfresults.png', 'df_act01.png', 'df_act01_0.png', 'df_act02.png', 'df_act02_0.png', 'df_screen01.png', 'df_screen02.png', 'diff.png', 'findshapes_1.png', 'findshapes_2.png', 'findsquare.png', 'fromjsonorg.png', 'from_nested_dict.png', 'generateimgs.png', 'horizontal.png', 'imagediffsquare.png', 'jsonstring.png', 'json_from_file.png', 'json_from_file_sp.png', 'json_url.png', 'lines.png', 'merg1.png', 'merg2.png', 'merg3.png', 'merg4.png', 'merg5.png', 'merg6.png', 'merg7.png', 'noformat.png', 'norm.png', 'pandasconsoleplott.png', 'pandsnesteddicthtml.png', 'papag_00000000.png', 'papag_00000001.png', 'papag_00000002.png', 'papag_00000003.png', 'papag_00000004.png', 'papag_00000005.png', 'papag_00000006.png', 'papag_00000007.png', 'papag_00000008.png', 'papag_00000009.png', 'pic1.png', 'pic2.png', 'pic3.png', 'pic4.png', 'pic5.png', 'PROXY.png', 'randomscreenshots.png', 'rect1.png', 'rect2.png', 'resultsscreenshots.png', 'rotate1.png', 'rotate2.png', 'screencap1.png', 'screencap2.png', 'screencap3.png', 'screencap4.png', 'screenshotpandasseries.png', 'screenshotpandastable.png', 'seleniumkeys.png', 'sendevent_key.png', 'sendevent_key_video.png', 'sift.png', 'splitted0.png', 'splitted1.png', 'templatematching1.png', 'templatematching2.png', 'templatematching3.png', 'templatematching4.png', 'templatematching5.png', 'templatematching6.png', 'templatematching7.png', 'tensorflowscreen.png', 'tesseractscreen.png', 'testwholescreenshot.png', 'timemea.png', 'vc01.png', 'vc01_0.png', 'vertical.png', 'zoompics_0000001.png', 'zoompics_0000002.png', 'zoompics_0000003.png', 'zoompics_0000004.png', 'zoompics_0000005.png', 'zoompics_0000006.png', 'zoompics_0000007.png', 'zoompics_0000008.png'], '.css': ['df_style.css'], '.html': ['fromjsonorg.html', 'from_nested_dict.html', 'jsonstring.html', 'json_from_file.html', 'json_url.html', 'myhtml.html'], '.jpeg': ['splitted1.jpeg'], '.mp4': ['tesseractscreen.mp4']}
######################################################
from a_cv_imwrite_imread_plus import open_image_in_cv
print(groupby_color_qty_in_region(
    im=open_image_in_cv(
        "https://github.com/hansalemaos/screenshots/raw/main/colorfind1.png"
    ),
    colors=[(0, 0, 0)],
    startx=None,
    starty=None,
    stopx=None,
    stopy=None,
    rect_w=10,
    rect_h=10,
    continue_on_exceptions=True,
    withindex=False,
    withvalue=True,
))
# output: {100: [(0, 0, 10, 10), (0, 10, 10, 20), (0, 20, 10, 30), (0, 30, 10, 40), (0, 40, 10, 50), (0, 50, 10, 60), (0, 60, 10, 70), (0, 70, 10, 80), (0, 80, 10, 90), (0, 90, 10, 100), (0, 100, 10, 110), (0, 110, 10, 120), (0, 120, 10, 130), (0, 130, 10, 140), (0, 140, 10, 150), (0, 150, 10, 160), (0, 160, 10, 170), (0, 170, 10, 180), (0, 180, 10, 190), (0, 190, 10, 200), (10, 0, 20, 10), (10, 10, 20, 20), (10, 20, 20, 30), (10, 30, 20, 40), (10, 40, 20, 50), (10, 50, 20, 60), (10, 60, 20, 70), (10, 70, 20, 80), (10, 80, 20, 90), (10, 90, 20, 100), (10, 100, 20, 110), (10, 110, 20, 120), (10, 120, 20, 130), (10, 130, 20, 140), (10, 140, 20, 150), (10, 150, 20, 160), (10, 160, 20, 170), (10, 170, 20, 180), (10, 180, 20, 190), (10, 190, 20, 200), (20, 0, 30, 10), (20, 10, 30, 20), (20, 20, 30, 30), (20, 30, 30, 40), (20, 40, 30, 50), (20, 50, 30, 60), (20, 60, 30, 70), (20, 70, 30, 80), (20, 80, 30, 90), (20, 90, 30, 100), (20, 100, 30, 110), (20, 110, 30, 120), (20, 120, 30, 130), (20, 130, 30, 140), (20, 140, 30, 150), (20, 150, 30, 160), (20, 160, 30, 170), (20, 170, 30, 180), (20, 180, 30, 190), (20, 190, 30, 200), (30, 0, 40, 10), (30, 10, 40, 20), (30, 20, 40, 30), (30, 30, 40, 40), (30, 40, 40, 50), (30, 50, 40, 60), (30, 60, 40, 70), (30, 130, 40, 140), (30, 140, 40, 150), (30, 150, 40, 160), (30, 160, 40, 170), (30, 170, 40, 180), (30, 180, 40, 190), (30, 190, 40, 200), (40, 0, 50, 10), (40, 10, 50, 20), (40, 20, 50, 30), (40, 30, 50, 40), (40, 40, 50, 50), (40, 50, 50, 60), (40, 140, 50, 150), (40, 150, 50, 160), (40, 160, 50, 170), (40, 170, 50, 180), (40, 180, 50, 190), (40, 190, 50, 200), (50, 0, 60, 10), (50, 10, 60, 20), (50, 20, 60, 30), (50, 30, 60, 40), (50, 40, 60, 50), (50, 50, 60, 60), (50, 140, 60, 150), (50, 150, 60, 160), (50, 160, 60, 170), (50, 170, 60, 180), (50, 180, 60, 190), (50, 190, 60, 200), (60, 0, 70, 10), (60, 10, 70, 20), (60, 20, 70, 30), (60, 160, 70, 170), (60, 170, 70, 180), (60, 180, 70, 190), (60, 190, 70, 200), (70, 0, 80, 10), (70, 10, 80, 20), (70, 20, 80, 30), (70, 170, 80, 180), (70, 180, 80, 190), (70, 190, 80, 200), (80, 0, 90, 10), (80, 10, 90, 20), (80, 20, 90, 30), (80, 170, 90, 180), (80, 180, 90, 190), (80, 190, 90, 200), (90, 0, 100, 10), (90, 10, 100, 20), (90, 20, 100, 30), (90, 170, 100, 180), (90, 180, 100, 190), (90, 190, 100, 200), (100, 0, 110, 10), (100, 10, 110, 20), (100, 20, 110, 30), (100, 170, 110, 180), (100, 180, 110, 190), (100, 190, 110, 200), (110, 0, 120, 10), (110, 10, 120, 20), (110, 20, 120, 30), (110, 170, 120, 180), (110, 180, 120, 190), (110, 190, 120, 200), (120, 0, 130, 10), (120, 10, 130, 20), (120, 20, 130, 30), (120, 170, 130, 180), (120, 180, 130, 190), (120, 190, 130, 200), (130, 0, 140, 10), (130, 10, 140, 20), (130, 20, 140, 30), (130, 30, 140, 40), (130, 160, 140, 170), (130, 170, 140, 180), (130, 180, 140, 190), (130, 190, 140, 200), (140, 0, 150, 10), (140, 10, 150, 20), (140, 20, 150, 30), (140, 30, 150, 40), (140, 40, 150, 50), (140, 50, 150, 60), (140, 140, 150, 150), (140, 150, 150, 160), (140, 160, 150, 170), (140, 170, 150, 180), (140, 180, 150, 190), (140, 190, 150, 200), (150, 0, 160, 10), (150, 10, 160, 20), (150, 20, 160, 30), (150, 30, 160, 40), (150, 40, 160, 50), (150, 50, 160, 60), (150, 140, 160, 150), (150, 150, 160, 160), (150, 160, 160, 170), (150, 170, 160, 180), (150, 180, 160, 190), (150, 190, 160, 200), (160, 0, 170, 10), (160, 10, 170, 20), (160, 20, 170, 30), (160, 30, 170, 40), (160, 40, 170, 50), (160, 50, 170, 60), (160, 60, 170, 70), (160, 130, 170, 140), (160, 140, 170, 150), (160, 150, 170, 160), (160, 160, 170, 170), (160, 170, 170, 180), (160, 180, 170, 190), (160, 190, 170, 200), (170, 0, 180, 10), (170, 10, 180, 20), (170, 20, 180, 30), (170, 30, 180, 40), (170, 40, 180, 50), (170, 50, 180, 60), (170, 60, 180, 70), (170, 70, 180, 80), (170, 80, 180, 90), (170, 90, 180, 100), (170, 100, 180, 110), (170, 110, 180, 120), (170, 120, 180, 130), (170, 130, 180, 140), (170, 140, 180, 150), (170, 150, 180, 160), (170, 160, 180, 170), (170, 170, 180, 180), (170, 180, 180, 190), (170, 190, 180, 200), (180, 0, 190, 10), (180, 10, 190, 20), (180, 20, 190, 30), (180, 30, 190, 40), (180, 40, 190, 50), (180, 50, 190, 60), (180, 60, 190, 70), (180, 70, 190, 80), (180, 80, 190, 90), (180, 90, 190, 100), (180, 100, 190, 110), (180, 110, 190, 120), (180, 120, 190, 130), (180, 130, 190, 140), (180, 140, 190, 150), (180, 150, 190, 160), (180, 160, 190, 170), (180, 170, 190, 180), (180, 180, 190, 190), (180, 190, 190, 200), (190, 0, 200, 10), (190, 10, 200, 20), (190, 20, 200, 30), (190, 30, 200, 40), (190, 40, 200, 50), (190, 50, 200, 60), (190, 60, 200, 70), (190, 70, 200, 80), (190, 80, 200, 90), (190, 90, 200, 100), (190, 100, 200, 110), (190, 110, 200, 120), (190, 120, 200, 130), (190, 130, 200, 140), (190, 140, 200, 150), (190, 150, 200, 160), (190, 160, 200, 170), (190, 170, 200, 180), (190, 180, 200, 190), (190, 190, 200, 200)], 65: [(30, 70, 40, 80)], 27: [(30, 80, 40, 90)], 11: [(30, 90, 40, 100)], 13: [(30, 100, 40, 110)], 30: [(30, 110, 40, 120), (60, 110, 70, 120), (60, 120, 70, 130), (100, 130, 110, 140), (110, 130, 120, 140), (120, 90, 130, 100), (120, 130, 130, 140), (130, 70, 140, 80), (130, 80, 140, 90)], 71: [(30, 120, 40, 130)], 76: [(40, 60, 50, 70), (130, 40, 140, 50), (160, 120, 170, 130)], 0: [(40, 70, 50, 80), (40, 80, 50, 90), (40, 90, 50, 100), (40, 100, 50, 110), (40, 110, 50, 120), (50, 70, 60, 80), (50, 80, 60, 90), (50, 90, 60, 100), (50, 100, 60, 110), (50, 110, 60, 120), (50, 120, 60, 130), (60, 70, 70, 80), (60, 80, 70, 90), (60, 90, 70, 100), (70, 40, 80, 50), (70, 50, 80, 60), (70, 70, 80, 80), (70, 80, 80, 90), (70, 110, 80, 120), (70, 120, 80, 130), (70, 130, 80, 140), (70, 140, 80, 150), (70, 150, 80, 160), (80, 40, 90, 50), (80, 50, 90, 60), (80, 70, 90, 80), (80, 80, 90, 90), (80, 110, 90, 120), (80, 120, 90, 130), (80, 130, 90, 140), (80, 140, 90, 150), (80, 150, 90, 160), (90, 40, 100, 50), (90, 50, 100, 60), (90, 70, 100, 80), (90, 80, 100, 90), (90, 110, 100, 120), (90, 120, 100, 130), (90, 130, 100, 140), (90, 140, 100, 150), (90, 150, 100, 160), (100, 40, 110, 50), (100, 50, 110, 60), (100, 60, 110, 70), (100, 70, 110, 80), (100, 80, 110, 90), (100, 110, 110, 120), (100, 120, 110, 130), (100, 140, 110, 150), (100, 150, 110, 160), (110, 40, 120, 50), (110, 50, 120, 60), (110, 60, 120, 70), (110, 70, 120, 80), (110, 80, 120, 90), (110, 110, 120, 120), (110, 120, 120, 130), (110, 140, 120, 150), (110, 150, 120, 160), (120, 40, 130, 50), (120, 50, 130, 60), (120, 60, 130, 70), (120, 70, 130, 80), (120, 80, 130, 90), (120, 100, 130, 110), (120, 110, 130, 120), (120, 120, 130, 130), (120, 140, 130, 150), (130, 100, 140, 110), (130, 110, 140, 120), (130, 120, 140, 130), (140, 70, 150, 80), (140, 80, 150, 90), (140, 90, 150, 100), (140, 100, 150, 110), (140, 110, 150, 120), (140, 120, 150, 130), (150, 70, 160, 80), (150, 80, 160, 90), (150, 90, 160, 100), (150, 100, 160, 110), (150, 110, 160, 120)], 1: [(40, 120, 50, 130), (120, 150, 130, 160), (150, 120, 160, 130)], 83: [(40, 130, 50, 140)], 60: [(50, 60, 60, 70), (60, 50, 70, 60), (60, 140, 70, 150), (140, 60, 150, 70)], 70: [(50, 130, 60, 140), (130, 50, 140, 60), (130, 140, 140, 150), (140, 130, 150, 140)], 97: [(60, 30, 70, 40)], 66: [(60, 40, 70, 50)], 52: [(60, 60, 70, 70)], 16: [(60, 100, 70, 110), (80, 30, 90, 40)], 51: [(60, 130, 70, 140)], 78: [(60, 150, 70, 160)], 43: [(70, 30, 80, 40)], 40: [(70, 60, 80, 70), (80, 60, 90, 70)], 6: [(70, 90, 80, 100)], 26: [(70, 100, 80, 110)], 72: [(70, 160, 80, 170)], 20: [(80, 90, 90, 100), (90, 90, 100, 100), (100, 90, 110, 100), (110, 90, 120, 100)], 10: [(80, 100, 90, 110), (90, 30, 100, 40), (90, 100, 100, 110), (100, 30, 110, 40), (100, 100, 110, 110)], 32: [(80, 160, 90, 170)], 36: [(90, 60, 100, 70)], 14: [(90, 160, 100, 170), (160, 90, 170, 100), (160, 100, 170, 110)], 15: [(100, 160, 110, 170)], 17: [(110, 30, 120, 40)], 9: [(110, 100, 120, 110), (130, 90, 140, 100)], 33: [(110, 160, 120, 170), (160, 110, 170, 120)], 54: [(120, 30, 130, 40), (130, 60, 140, 70)], 73: [(120, 160, 130, 170)], 58: [(130, 130, 140, 140)], 81: [(130, 150, 140, 160), (150, 130, 160, 140)], 69: [(150, 60, 160, 70)], 64: [(160, 70, 170, 80)], 31: [(160, 80, 170, 90)]}
######################################################
print(groupby_even_odd(
    seq=[1, 2, 3, 45, 56, 6, 32, 12], continue_on_exceptions=True, withindex=False
))
# output: {'odd': [1, 3, 45], 'even': [2, 56, 6, 32, 12]}
######################################################
print(
    groupby_files_folder_link(
        seq=[
            os.path.join(r"F:\gitrep\screenshots", x)
            for x in os.listdir(r"F:\gitrep\screenshots")
        ],
        continue_on_exceptions=True,
        withindex=False,
        withvalue=True,
    )
)
# output: {'folder': ['F:\\gitrep\\screenshots\\.git', 'F:\\gitrep\\screenshots\\color', 'F:\\gitrep\\screenshots\\debugdf', 'F:\\gitrep\\screenshots\\enumeratefiles', 'F:\\gitrep\\screenshots\\sortfiles'], 'file': ['F:\\gitrep\\screenshots\\.gitignore', 'F:\\gitrep\\screenshots\\144.png', 'F:\\gitrep\\screenshots\\146.png', 'F:\\gitrep\\screenshots\\2022-12-27 03_40_27-.png', 'F:\\gitrep\\screenshots\\2022-12-27 03_49_27-.png', 'F:\\gitrep\\screenshots\\2022-12-27 04_01_57-.png', 'F:\\gitrep\\screenshots\\2022-12-27 04_02_09-.png', 'F:\\gitrep\\screenshots\\2023-01-01 09_48_27-single_elements.png', 'F:\\gitrep\\screenshots\\add_methods.png', 'F:\\gitrep\\screenshots\\allcolors.png', 'F:\\gitrep\\screenshots\\asciiart.png', 'F:\\gitrep\\screenshots\\bluestacksauto.png', 'F:\\gitrep\\screenshots\\buttonsearch.png', 'F:\\gitrep\\screenshots\\collage.png', 'F:\\gitrep\\screenshots\\colorfind1.png', 'F:\\gitrep\\screenshots\\colorfind1xxx.png', 'F:\\gitrep\\screenshots\\colorfind2.png', 'F:\\gitrep\\screenshots\\colorfind3.png', 'F:\\gitrep\\screenshots\\colorfind4.png', 'F:\\gitrep\\screenshots\\cv2_putTrueTypeText_000000.png', 'F:\\gitrep\\screenshots\\cv2_putTrueTypeText_000001.png', 'F:\\gitrep\\screenshots\\cv2_putTrueTypeText_000002.png', 'F:\\gitrep\\screenshots\\cv2_putTrueTypeText_000003.png', 'F:\\gitrep\\screenshots\\cv2_putTrueTypeText_000004.png', 'F:\\gitrep\\screenshots\\cv2_putTrueTypeText_000005.png', 'F:\\gitrep\\screenshots\\cv2_putTrueTypeText_000006.png', 'F:\\gitrep\\screenshots\\cv2_putTrueTypeText_000007.png', 'F:\\gitrep\\screenshots\\cv2_putTrueTypeText_000008.png', 'F:\\gitrep\\screenshots\\cv2_putTrueTypeText_000009.png', 'F:\\gitrep\\screenshots\\cv2_putTrueTypeText_000010.png', 'F:\\gitrep\\screenshots\\cv2_putTrueTypeText_000011.png', 'F:\\gitrep\\screenshots\\cv2_putTrueTypeText_000012.png', 'F:\\gitrep\\screenshots\\cv2_putTrueTypeText_000013.png', 'F:\\gitrep\\screenshots\\cv2_putTrueTypeText_000014.png', 'F:\\gitrep\\screenshots\\cv2_putTrueTypeText_000015.png', 'F:\\gitrep\\screenshots\\cv2_putTrueTypeText_000017.png', 'F:\\gitrep\\screenshots\\dfresults.png', 'F:\\gitrep\\screenshots\\df_act01.png', 'F:\\gitrep\\screenshots\\df_act01_0.png', 'F:\\gitrep\\screenshots\\df_act02.png', 'F:\\gitrep\\screenshots\\df_act02_0.png', 'F:\\gitrep\\screenshots\\df_screen01.png', 'F:\\gitrep\\screenshots\\df_screen02.png', 'F:\\gitrep\\screenshots\\df_style.css', 'F:\\gitrep\\screenshots\\diff.png', 'F:\\gitrep\\screenshots\\findshapes_1.png', 'F:\\gitrep\\screenshots\\findshapes_2.png', 'F:\\gitrep\\screenshots\\findsquare.png', 'F:\\gitrep\\screenshots\\fromjsonorg.html', 'F:\\gitrep\\screenshots\\fromjsonorg.png', 'F:\\gitrep\\screenshots\\from_nested_dict.html', 'F:\\gitrep\\screenshots\\from_nested_dict.png', 'F:\\gitrep\\screenshots\\generateimgs.png', 'F:\\gitrep\\screenshots\\horizontal.png', 'F:\\gitrep\\screenshots\\imagediffsquare.png', 'F:\\gitrep\\screenshots\\jsonstring.html', 'F:\\gitrep\\screenshots\\jsonstring.png', 'F:\\gitrep\\screenshots\\json_from_file.html', 'F:\\gitrep\\screenshots\\json_from_file.png', 'F:\\gitrep\\screenshots\\json_from_file_sp.png', 'F:\\gitrep\\screenshots\\json_url.html', 'F:\\gitrep\\screenshots\\json_url.png', 'F:\\gitrep\\screenshots\\LICENSE', 'F:\\gitrep\\screenshots\\lines.png', 'F:\\gitrep\\screenshots\\merg1.png', 'F:\\gitrep\\screenshots\\merg2.png', 'F:\\gitrep\\screenshots\\merg3.png', 'F:\\gitrep\\screenshots\\merg4.png', 'F:\\gitrep\\screenshots\\merg5.png', 'F:\\gitrep\\screenshots\\merg6.png', 'F:\\gitrep\\screenshots\\merg7.png', 'F:\\gitrep\\screenshots\\myhtml.html', 'F:\\gitrep\\screenshots\\noformat.png', 'F:\\gitrep\\screenshots\\norm.png', 'F:\\gitrep\\screenshots\\pandasconsoleplott.png', 'F:\\gitrep\\screenshots\\pandsnesteddicthtml.png', 'F:\\gitrep\\screenshots\\papag_00000000.png', 'F:\\gitrep\\screenshots\\papag_00000001.png', 'F:\\gitrep\\screenshots\\papag_00000002.png', 'F:\\gitrep\\screenshots\\papag_00000003.png', 'F:\\gitrep\\screenshots\\papag_00000004.png', 'F:\\gitrep\\screenshots\\papag_00000005.png', 'F:\\gitrep\\screenshots\\papag_00000006.png', 'F:\\gitrep\\screenshots\\papag_00000007.png', 'F:\\gitrep\\screenshots\\papag_00000008.png', 'F:\\gitrep\\screenshots\\papag_00000009.png', 'F:\\gitrep\\screenshots\\pic1.png', 'F:\\gitrep\\screenshots\\pic2.png', 'F:\\gitrep\\screenshots\\pic3.png', 'F:\\gitrep\\screenshots\\pic4.png', 'F:\\gitrep\\screenshots\\pic5.png', 'F:\\gitrep\\screenshots\\PROXY.png', 'F:\\gitrep\\screenshots\\randomscreenshots.png', 'F:\\gitrep\\screenshots\\rect1.png', 'F:\\gitrep\\screenshots\\rect2.png', 'F:\\gitrep\\screenshots\\resultsscreenshots.png', 'F:\\gitrep\\screenshots\\rotate1.png', 'F:\\gitrep\\screenshots\\rotate2.png', 'F:\\gitrep\\screenshots\\screencap1.png', 'F:\\gitrep\\screenshots\\screencap2.png', 'F:\\gitrep\\screenshots\\screencap3.png', 'F:\\gitrep\\screenshots\\screencap4.png', 'F:\\gitrep\\screenshots\\screenshotpandasseries.png', 'F:\\gitrep\\screenshots\\screenshotpandastable.png', 'F:\\gitrep\\screenshots\\seleniumkeys.png', 'F:\\gitrep\\screenshots\\sendevent_key.png', 'F:\\gitrep\\screenshots\\sendevent_key_video.png', 'F:\\gitrep\\screenshots\\sift.png', 'F:\\gitrep\\screenshots\\splitted0.png', 'F:\\gitrep\\screenshots\\splitted1.jpeg', 'F:\\gitrep\\screenshots\\splitted1.png', 'F:\\gitrep\\screenshots\\templatematching1.png', 'F:\\gitrep\\screenshots\\templatematching2.png', 'F:\\gitrep\\screenshots\\templatematching3.png', 'F:\\gitrep\\screenshots\\templatematching4.png', 'F:\\gitrep\\screenshots\\templatematching5.png', 'F:\\gitrep\\screenshots\\templatematching6.png', 'F:\\gitrep\\screenshots\\templatematching7.png', 'F:\\gitrep\\screenshots\\tensorflowscreen.png', 'F:\\gitrep\\screenshots\\tesseractscreen.mp4', 'F:\\gitrep\\screenshots\\tesseractscreen.png', 'F:\\gitrep\\screenshots\\testwholescreenshot.png', 'F:\\gitrep\\screenshots\\timemea.png', 'F:\\gitrep\\screenshots\\vc01.png', 'F:\\gitrep\\screenshots\\vc01_0.png', 'F:\\gitrep\\screenshots\\vertical.png', 'F:\\gitrep\\screenshots\\zoompics_0000001.png', 'F:\\gitrep\\screenshots\\zoompics_0000002.png', 'F:\\gitrep\\screenshots\\zoompics_0000003.png', 'F:\\gitrep\\screenshots\\zoompics_0000004.png', 'F:\\gitrep\\screenshots\\zoompics_0000005.png', 'F:\\gitrep\\screenshots\\zoompics_0000006.png', 'F:\\gitrep\\screenshots\\zoompics_0000007.png', 'F:\\gitrep\\screenshots\\zoompics_0000008.png']}
######################################################
import pandas as pd
import math
import numpy as np
print(groupby_sys_size(seq=[pd.NA, np.nan, None, math.nan, 3,54,3,(22,34,412), {323,31}, [3312,3]], continue_on_exceptions=True, withindex=False))

# output: {48: [<NA>], 24: [nan, nan], 16: [None], 28: [3, 54, 3], 64: [(22, 34, 412)], 216: [{323, 31}], 72: [[3312, 3]]}
######################################################
print(groupby_first_item(
    seq=[[1,2,34,], (1,32,4), (2,3,4,54), [2,3,3]], continue_on_exceptions=True, withindex=False
))
# output: {1: [[1, 2, 34], (1, 32, 4)], 2: [(2, 3, 4, 54), [2, 3, 3]]}
######################################################
print(groupby_words_in_texts(
    textlist=["autobahn", "computerproblem", "kind", "opa", "kind opa"],
    wordlist=["kind", "opa"],
    case_sen=False,
    continue_on_exceptions=True,
    withindex=False,
    boundary_right=True,
    boundary_left=True,
))
# output: {(): ['autobahn', 'computerproblem'], ('kind',): ['kind'], ('opa',): ['opa'], ('kind', 'opa'): ['kind opa']}
######################################################
print(groupby_sum(
    seq=[[1, 2, 2], [5], [2, 3], [4, 4, 4], [12, 0], [6, 6], [1, 2]],
    continue_on_exceptions=True,
    withindex=False,
))
# output: {5: [[1, 2, 2], [5], [2, 3]], 12: [[4, 4, 4], [12, 0], [6, 6]], 3: [[1, 2]]}
######################################################
print(groupby_valid_url(
    seq=["https://www.google.com", "google.com/", 'bababa', 'http://baba.de'],
    continue_on_exceptions=True,
    withindex=False,
    withvalue=True,
))
# output: {'valid': ['https://www.google.com', 'http://baba.de'], 'not_valid': ['google.com/', 'bababa']}
######################################################
print(groupby_literal_eval_type(
    seq=['11', 'bb', '"bb"'], continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {<class 'int'>: ['11'], 'EXCEPTION: malformed node or string: <ast.Name object at 0x0000000013A54340>': ['bb'], <class 'str'>: ['"bb"']}
######################################################
rint(groupby_decoding_result(bytes_=b'\\U0001D11E',mode='strict',
     continue_on_exceptions=True, withindex=False, withvalue=True
))
# output:  {'\\U0001D11E': ['ascii',
#                 'big5',
#                 'big5hkscs',
#                 'charmap',
# 				 ......
#                 'palmos',
#                 'ptcp154',
#                 'shift_jis',
#                 'tis_620',
#                 'utf_7',
#                 'utf_8',
#                 'utf_8_sig'],
# '¥U0001D11E': ['shift_jisx0213', 'shift_jis_2004'],
# '啜〰\u3130ㅄ䔱': ['utf_16', 'utf_16_le'],
# '展〰〱䐱ㅅ': ['utf_16_be'],
# '𝄞': ['raw_unicode_escape', 'unicode_escape']}
######################################################
print(groupby_percentage(
    seq=10*[1,2,34,4,2,3,54,6,6,4,3,2,21,45,56], percent_true=63.7, continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {False: [1, 2, 4, 3, 2, 34, 4, 2, 3, 4, 56, 4, 2, 45, 2, 2, 3, 54, 2, 21, 1, 4, 6, 3, 21, 34, 2, 54, 3, 21, 45, 1, 2, 4, 54, 6, 4, 56, 1, 4, 56, 1, 2, 4, 54, 3, 1, 54, 4, 3, 21], True: [2, 34, 4, 3, 54, 6, 6, 21, 45, 56, 1, 2, 54, 6, 6, 3, 2, 21, 45, 1, 2, 34, 2, 3, 54, 6, 6, 4, 3, 21, 56, 1, 34, 4, 6, 6, 4, 3, 45, 56, 2, 34, 2, 3, 54, 6, 4, 2, 45, 56, 1, 2, 4, 3, 6, 6, 4, 2, 56, 34, 2, 3, 6, 3, 2, 21, 45, 2, 34, 2, 3, 54, 6, 6, 4, 3, 2, 21, 45, 34, 2, 3, 6, 6, 4, 2, 21, 45, 56, 2, 34, 4, 2, 3, 6, 6, 2, 45, 56]}
######################################################
print(groupby_almost_equal(
    seq=[11,200,34,4,52,63,54,65,67,48,3,2,21,55,56,59,61,60], value=60, equallimit=3, continue_on_exceptions=True, withindex=False, withvalue=True
))
# output: {False: [11, 200, 34, 4, 52, 54, 65, 67, 48, 3, 2, 21, 55, 56], True: [63, 59, 61, 60]}
######################################################
coordlist = 2 * [(1, 2), (34, 4), (2, 3), (61, 60)]
print(groupby_coords_almost_equal(seq=coordlist, x_coord=4, y_coord=3, limit_x=5, limit_y=1,
                                  continue_on_exceptions=True, withindex=False, withvalue=True))

# output: {(True, True): [(1, 2), (2, 3), (1, 2), (2, 3)], (False, True): [(34, 4), (34, 4)], (False, False): [(61, 60), (61, 60)]}
######################################################
coordlist = [(745, 519),
 (747, 522),
 (747, 517),
 (747, 517),
 (750, 522),
 (756, 449),
 (757, 461),
 (757, 461),
 (757, 438),
 (830, 144),
 (759, 435),
 (759, 435),
 (761, 468),
 (761, 468),
 (764, 521),
 (1079, 199),
 (770, 474),
 (770, 425),
 (773, 516),
 (776, 515),
 (776, 515),
 (778, 520),
 (779, 519),
 (780, 420),
 (780, 420),
 (782, 478),
 (782, 478),
 (1083, 151),
 (1083, 151),
 (1083, 151),
 (1083, 151),
 (784, 478),
 (759, 435),
 (784, 478),
 (819, 137),
 (819, 137),
 (819, 137),
 (797, 524),
 (825, 125),
 (826, 149),
 (800, 446),
 (800, 446),
 (801, 517),
 (801, 517),
 (802, 520),
 (802, 520),
 (804, 519),
 (804, 519),
 (808, 431),
 (808, 431),
 (809, 464),
 (809, 464),
 (812, 438),
 (813, 449)]

xx=(group_coordinates_by_distance(coordlist, limit_x=10, limit_y=10, continue_on_exceptions=True))
for x in xx:
    print(x)

# output: ((813, 449),)
# ((779, 519), (773, 516), (776, 515), (778, 520), (764, 521))
# ((808, 431), (812, 438))
# ((1083, 151),)
# ((830, 144), (826, 149))
# ((761, 468), (757, 461), (770, 474))
# ((825, 125),)
# ((756, 449),)
# ((745, 519), (747, 517), (750, 522), (747, 522))
# ((780, 420), (770, 425))
# ((784, 478), (782, 478))
# ((804, 519), (802, 520), (797, 524), (801, 517))
# ((757, 438), (759, 435))
# ((819, 137),)
# ((809, 464),)
# ((800, 446),)
# ((1079, 199),)
######################################################
```



### Iter stuff

```python
######################################################
for _ in iter_enumerate_multiple([3, 4], [55, 55]):
    print(_)

# output: 
(0, 3, 55)
(1, 4, 55)
######################################################	
for j in iter__chain([3, 4], [55, 55], [{33:31}, 4,1]):
    print(j)	

# output: 
3
4
55
55
{33: 31}
4
1	
######################################################
people = {1: {"name": "John", "age": "27", "sex": "Male"}, 2: {"name": "Marie", "age": "22", "sex": "Female"},
          3: {"name": "Luna", "age": "24", "sex": "Female"},
          4: {"name": "Peter", "age": "29", "sex": ["Female", "Male"], 1: "xx", "sex2": ("Female", "Male"), }, }
for keys, item in iter_nested_dict(people):
    print(keys, item)

# output: 
(1, 'name') John
(1, 'age') 27
(1, 'sex') Male
(2, 'name') Marie
(2, 'age') 22
(2, 'sex') Female
(3, 'name') Luna
(3, 'age') 24
(3, 'sex') Female
(4, 'name') Peter
(4, 'age') 29
(4, 'sex', 0) Female
(4, 'sex', 1) Male
(4, 1) xx
(4, 'sex2', 0) Female
(4, 'sex2', 1) Male
######################################################
for x in (iter_split_list_into_chunks_fill_rest(n=3, iterable=[1, 2, 3, 4, 4, 2, 21], fillvalue=None)):
    print(x)
	
# output: 	
(1, 2, 3)
(4, 4, 2)
(21, None, None)
######################################################
print(
    list(
        iter_cycle_shortest(
            iter1=[33, 133, 35, 3300],
            iter2=[331, 1133, 434, 44, 555, 22, 767, 446, 66, 2234],
        )
    )
)
# output: 
# [(33, 331), (133, 1133), (35, 434), (3300, 44), (33, 555), (133, 22), (35, 767), (3300, 446), (33, 66), (133, 2234)]

print(
    list(
        iter_cycle_shortest(
            iter1=[331, 1133, 434, 44, 555, 22, 767, 446, 66, 2234],
            iter2=[33, 133, 35, 3300],
        )
    )
)
# output: 
# [(331, 33), (1133, 133), (434, 35), (44, 3300), (555, 33), (22, 133), (767, 35), (446, 3300), (66, 33), (2234, 133)]
######################################################
print(list(iter_reverse_lists_of_list(lis=((323,33,331), (331,552,1)))))
# output: [[331, 33, 323], [1, 552, 331]]
######################################################
print(list(iter_split_by_index("For performance reasons, the value of errors is not checked for validity", [6, 8, 12, 13, 18])))
# output: ['For pe', 'rf', 'orma', 'n', 'ce re', 'asons, the value of errors is not checked for validity']

print(list(iter_split_by_index([32.,3,123,424,4242,4,24,532,54,53254,24,24,24,532,35624,42], [6, 8, 12, 13, 18])))
# output: [[32.0, 3, 123, 424, 4242, 4], [24, 532], [54, 53254, 24, 24], [24], [532, 35624, 42], []]
######################################################
for kl in iter_get_every_nth_element(iterable=[1,2,3,4,4,5,6,3,2.4,4,5,5,4],start=0, step=2):
    print(kl)
# output: 
1
3
4
6
2.4
5
4
######################################################
print(list(iter_find_same_ending_elements(iters=[list((1,2,3,4)), list((1,2,3,4)), list((5,1,2,3,4))])))
# output: [1, 2, 3, 4]
######################################################
print(list(iter_find_same_beginning_elements(iters=[list((1,2,3,4)), list((1,2,3,4)), list((1,2,3,4,5))])))
# output: [1, 2, 3, 4]
######################################################
for x in iter_nested_list_from_item(variable=[244, 244, 2], shape=(2, 3,2)):
    print(x)
# output: [[[244, 244, 2], [244, 244, 2]], [[244, 244, 2], [244, 244, 2]], [[244, 244, 2], [244, 244, 2]]]
#         [[[244, 244, 2], [244, 244, 2]], [[244, 244, 2], [244, 244, 2]], [[244, 244, 2], [244, 244, 2]]]
######################################################
print(list(iter_repeat_items(iterable=[34,4,23], reps=3)))
# output: [34, 34, 34, 4, 4, 4, 23, 23, 23]
######################################################
print(iter_count_occurrences_of_item_in_multiple_iterables(iterables=[['hallo', 'mein', 'Freund', 'ou'],['hallo', 'mein', 'Freund', 'oxu']]))
# output: Counter({'hallo': 2, 'mein': 2, 'Freund': 2, 'ou': 1, 'oxu': 1})
######################################################
print(iter_count_occurrences_of_item_in_multiple_iterables_flatten_everything(iterables=[['hallo', ['mein', 'Freund'], 'ou'],['hallo', ['mein'], 'Freund', 'oxu']]))
# output: Counter({'hallo': 2, 'mein': 2, 'Freund': 2, 'ou': 1, 'oxu': 1})
######################################################
list1 = [(255, 255, 0), (255, 21, 23), (0, 0, 0), (1, 1, 1)]
list2 = ['yellow', 'red', 'black', 'black']
print(list(iter_values_from_list1_based_on_values_of_list2(list1, list2, condition=lambda x: x == 'black')))
# output: [(0, 0, 0), (1, 1, 1)]
######################################################
print(list(iter_windowed(iterable=[1,23,3,2,34], n=2)))
# output: [(1, 23), (23, 3), (3, 2), (2, 34)]
######################################################
for subit in iter_batch('ABCDEFG', 3):
    for x in subit:
        print(x, end=' ')
    print()

# output:	
A B C 
D E F 
G 
######################################################
npar = numpy_all_permutations(iterable=list('abcdefghijklmnop'), len_of_each=6)
# output:
array([['a', 'b', 'c', 'd', 'e', 'f'],
       ['a', 'b', 'c', 'd', 'e', 'g'],
       ['a', 'b', 'c', 'd', 'e', 'h'],
       ...,
       ['p', 'o', 'n', 'm', 'l', 'i'],
       ['p', 'o', 'n', 'm', 'l', 'j'],
       ['p', 'o', 'n', 'm', 'l', 'k']], dtype='<U1')

print(npar.shape)
(5765760, 6)
######################################################
print(numpy_random_array_no_reps_beginning_end(iterable=['hallo','du','wie','er', 'ich', 'es'], block_count=3))
# output:
[['du' 'wie' 'es' 'ich' 'er' 'hallo']
 ['es' 'hallo' 'ich' 'du' 'wie' 'er']
 ['du' 'ich' 'wie' 'er' 'es' 'hallo']]	
######################################################
l1 = [1, 2, 3]
l2 = [4, 5, 6]
l3 = [7, 8, 9]
for l11, l22, l33 in iter_nested_for_loop(l1, l2, l3):
    print(l11, l22, l33)
# output:
1 4 7
1 4 8
1 4 9
1 5 7
1 5 8
1 5 9
1 6 7
1 6 8
1 6 9
2 4 7
2 4 8
2 4 9
2 5 7
2 5 8
2 5 9
2 6 7
2 6 8
2 6 9
3 4 7
3 4 8
3 4 9
3 5 7
3 5 8
3 5 9
3 6 7
3 6 8
3 6 9
######################################################
l1 = [1, 2, 3]
l2 = [4, 5, 6]
l3 = [7, 8, 9]
for ini, l11, l22, l33 in iter_nested_for_loop_enumerate(l1, l2, l3):
    print(ini, l11, l22, l33)


# output:

0 1 4 7
1 1 4 8
2 1 4 9
3 1 5 7
4 1 5 8
5 1 5 9
6 1 6 7
7 1 6 8
8 1 6 9
9 2 4 7
10 2 4 8
11 2 4 9
12 2 5 7
13 2 5 8
14 2 5 9
15 2 6 7
16 2 6 8
17 2 6 9
18 3 4 7
19 3 4 8
20 3 4 9
21 3 5 7
22 3 5 8
23 3 5 9
24 3 6 7
25 3 6 8
26 3 6 9
######################################################
for i in iter_chain_flatten([332313,5434,2342], {3232,342},312, 331):
    print(i)

# output:	
332313
5434
2342
3232
342
312
331

######################################################
for ini,cir in enumerate(iter_spiral(radius=10,eccentricity = 1.5,step = 0.1,t = 0)):
    print(cir)
    if ini==15:
        break

# output:
(1.4925062479170388, 0.09983341664682815)
(2.940199733523725, 0.39733866159012243)
(4.299014201065228, 0.8865606199840189)
(5.526365964017311, 1.557673369234602)
(6.581869214177796, 2.397127693021015)
(7.428020534187105, 3.3878548403702125)
(8.030842966487128, 4.509523810663837)
(8.360480512165985, 5.7388487271961806)
(8.39173457165397, 7.04994218664735)
(8.104534588022096, 8.414709848078962)
(7.484336003522027, 9.803280960675787)
(6.522439580580125, 11.184469031606715)
(5.2162271581794535, 12.526256410423509)
(3.5693100009050576, 13.796296219838446)
(1.5915870375233105, 14.962424799060818)
(-0.700788535230937, 15.993177648664085)		

######################################################
vara= ['key1','x', 'key2',['one', 'two', 'three']]

# normal python version
for k in vara:
    if isinstance(k,list):
        for li in k:
            print(k, li)
    else:
        print(k)

# output:	
key1
x
key2
['one', 'two', 'three'] one
['one', 'two', 'three'] two
['one', 'two', 'three'] three


# same thing, but much shorter:
for k in iter_nested(iterable=vara):
    print(k)

# output:		
['key1']
['x']
['key2']
[['one', 'two', 'three'], 'one']
[['one', 'two', 'three'], 'two']
[['one', 'two', 'three'], 'three']	

######################################################
# same as iter_nested, but with path to each value
for k in iter_nested_with_path(iterable=vara):
    print(k)

# output:	
[((0,), 'key1')]
[((1,), 'x')]
[((2,), 'key2')]
[((3,), ['one', 'two', 'three']), ((3, 0), 'one')]
[((3,), ['one', 'two', 'three']), ((3, 1), 'two')]
[((3,), ['one', 'two', 'three']), ((3, 2), 'three')]	
######################################################
print(list(iter_add_one_item_each_iteration(iterable=vara)))
# output:	[['key1'], ['key1', 'x'], ['key1', 'x', 'key2'], ['key1', 'x', 'key2', ['one', 'two', 'three']]]
######################################################
print(list(iter_add_one_item_each_iteration_reverse(iterable=vara)))
# output:	[[['one', 'two', 'three']], ['key2', ['one', 'two', 'three']], ['x', 'key2', ['one', 'two', 'three']], ['key1', 'x', 'key2', ['one', 'two', 'three']]]
######################################################



		
```
