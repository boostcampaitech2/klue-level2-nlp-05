import sys

import fasttext
import fasttext.util

def main(LOAD_DIR, SAVE_DIR, to_dims: int=100):
    ft = fasttext.load_model(LOAD_DIR)
    fasttext.util.reduce_model(ft, to_dims)
    ft.save_model(SAVE_DIR)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))