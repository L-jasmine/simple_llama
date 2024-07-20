# Simple Llama

This project, Simple Llama, is a library that encapsulates commonly used large model prompts based on the llama-cpp-2 framework. It aims to simplify the interaction with large-scale models by providing a streamlined interface for managing and invoking model prompts. This library is designed to enhance the efficiency and ease of use for developers working with large models in various applications.

##  Clone the repository
```
git clone https://github.com/L-jasmine/simple_llama
```

## Download Llama model.
```
wget https://huggingface.co/second-state/Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf
```

## Configure Environment Variables
This project uses dynamic linking to connect to llama.cpp, so it is necessary to download or compile the llama.cpp dynamic link library in advance.

Before running the project, you need to configure environment variables to specify the location of the Llama library and the search path for dynamic link libraries. Please follow the steps below:

```shell
export LLAMA_LIB={LLama_Dynamic_Library_Dir}
# export LD_LIBRARY_PATH={LLama_Dynamic_Library_Dir}
```

## Run the Example

Use the following command to run the example program:

```shell
cargo run --example simple -- --model-path Meta-Llama-3-8B-Instruct-Q5_K_M.gguf --model-type llama3 --prompt-path static/prompt.example.toml
```

## Contributions

We welcome any form of contributions, including bug reports, new feature suggestions, and code submissions.

## License

This project is licensed under the MIT License.
