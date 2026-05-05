import argparse

from nanovllm.config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="配置读取")
    args = parser.parse_args()
    config = Config(model=args.model)
    print(config)


if __name__ == "__main__":
    main()
