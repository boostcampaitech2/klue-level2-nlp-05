import fasttext
import fasttext.util

if __name__ == '__main__':
    fasttext.util.download_model("ko", if_exists="ignore")
    # this takes more than 10 mins