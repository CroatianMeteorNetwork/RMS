import math


def positive(x):
    return int((abs(x) + x)/2)


def cyclicSubsequence(a, b, unknowns=False):
    """ Checks if a is a cyclic subsequence of b """
    return next(_cyclicSubsequence(a, b, unknowns))


def _cyclicSubsequence(a, b, unknowns=False):
    """ Returns a generator that checks if a is a cyclic subsequence of b """
    if not unknowns:
        for i in range(len(b)):
            if b[i:i + len(a)] + b[positive(i - len(b)):positive(i - len(b) + len(a))] == a:
                yield i
    else:
        for i in range(len(b)):
            found = True
            for j in range(len(a)):
                if b[i + j - len(b)] != a[j] and a[j] is not None:
                    found = False
                    break
            if found:
                yield i
    yield None


def subsequence(a, b):
    """ Checks if a is a subsequence of b """

    for i in range(len(b)):
        if b[i:i + len(a)] == a:
            return i
    return None


def findInDeBruijnSequence(test, sequence, reverse=None, unknowns=False):
    """

    Arguments:
        test:
        sequence:
        reverse:
        unknowns: if True, None will be considered a correct value

    Returns:
        None: if sequence cannot be found


    """
    if len(test) < int(math.log(len(sequence), 2)):
        return None, None

    if reverse is False:
        return cyclicSubsequence(test, sequence, unknowns=unknowns), None
    elif reverse is True:
        return None, cyclicSubsequence(test, sequence[::-1], unknowns=unknowns)
    else:
        forward = cyclicSubsequence(test, sequence, unknowns=unknowns)
        backward = cyclicSubsequence(test, sequence[::-1], unknowns=unknowns)
        return forward, backward


def findAllInDeBruijnSequence(test, sequence, reverse=None, unknowns=False):
    if reverse is False:
        return list(_cyclicSubsequence(test, sequence, unknowns=unknowns))[:-1], []

    elif reverse is True:
        return [], list(_cyclicSubsequence(test, sequence[::-1], unknowns=unknowns))[:-1]

    else:
        forward = list(_cyclicSubsequence(test, sequence, unknowns=unknowns))[:-1]
        backward = list(_cyclicSubsequence(test, sequence[::-1], unknowns=unknowns))[:-1]
        return forward, backward


def generateDeBruijnSequence(k, n):
    """
    Howie, R., Paxman, J., Bland, P., Towner, M., Sansom, E, & Devillpix, H. (2017). Submillisecond
    fireball timing using de Bruijn timecodes. pp. 1675

    Arguments:
        k: [int] number of characters in alphabet (eg. 2 for {0,1}, 3 for {0,1,2})
        n: [int] subsequence length

    Return: [list] generated de brujn sequence

    """
    sequence = [0]*n
    while len(sequence) < k**n:
        i = k - 1
        while True:
            test = sequence[-(n - 1):] + [i]
            if subsequence(test, sequence) is None:
                sequence.append(i)
                break
            else:
                i -= 1
    return sequence


if __name__ == '__main__':

    
    sequence = generateDeBruijnSequence(2, 9)
    print(''.join([str(x) for x in sequence]))

    #print(''.join([str(x) for x in sequence[::-1]]))

    # print(cyclicSubsequence([1, 2, None, 4, None, 5],
    #                         [5, 1, 2, 1, 4, 1], unknowns=True))
    #
    # n = 9
    # k = 2
    # for n in range(2, 11):
    #     sequence = generateDeBruijnSequence(k, n)
    #     reverse_sequence = sequence[::-1]
    #
    #     i = 0
    #     l = [cyclicSubsequence(sequence[j:j + n + i], reverse_sequence) is not None for j in
    #          range(len(sequence) - n - i + 1)]
    #
    #     while any(l):
    #         print(n, i + n, sum(l)/len(l)*100)
    #         i += 1
    #         l = [cyclicSubsequence(sequence[j:j + n + i], reverse_sequence) is not None for j in
    #              range(len(sequence) - n - i + 1)]
    #
    #     print(n, i + n, sum(l))
    #     print('-'*100)
