import string
import random
extra = '      '
symbols = '.-./-'


def id_generator(size=6, chars=string.ascii_uppercase + string.digits + string.ascii_lowercase + symbols):
    return ''.join(random.choice(chars) for _ in range(size))


# print id_generator()


def writer(n):
    file = open('bigram_wordlist.txt', 'w+')
    for i in range(n):
        file.write( id_generator() + ' ' + id_generator() + '\n')
    file.close()


if __name__ == '__main__':
    writer(86496)
