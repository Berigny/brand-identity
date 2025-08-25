import os
from dotenv import load_dotenv
from app.utils.llm_factory import make_llm, with_retry


def main():
    if os.path.exists(".env.local"):
        load_dotenv(".env.local")
    else:
        load_dotenv()

    llm = make_llm()
    resp = with_retry(lambda: llm.invoke("Hello from factory!"))
    print(resp)


if __name__ == "__main__":
    main()

