
def cal_length(ax):

    # ax is list of int
    ls = []
    temp = ax[0]
    li = -1

    for ai in range(len(ax)):
        if temp == ax[ai]:
            li += 1
        else:
            temp = ax[ai]
            li += 1
            ls.append(li)
            li = 0
    ls.append(li + 1)
    return ls