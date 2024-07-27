use clap::{Parser, ValueEnum};
use std::{collections::HashMap, io::Write, num::NonZeroU32, sync::Arc};

use simple_llama::{
    llm::{self, Content, LlamaContextParams, LlamaModelParams},
    ChatRequest,
};

#[derive(Debug, Parser)]
struct Args {
    /// Path to the model
    #[arg(short, long)]
    model_path: String,

    /// path to the prompt
    #[arg(short, long)]
    prompt_path: String,

    /// Type of the model
    #[arg(short('t'), long, value_enum)]
    model_type: ModelType,

    /// full prompt chat
    #[arg(long)]
    full_chat: bool,
}

#[derive(Debug, Clone, ValueEnum)]
enum ModelType {
    Llama3,
    Hermes2ProLlama3,
    Gemma2,
    Qwen,
}

mod prompt_template {
    use simple_llama::PromptTemplate;

    pub fn llama3_prompt_template() -> PromptTemplate {
        PromptTemplate {
            header_prefix: "<|start_header_id|>".to_string(),
            header_suffix: "<|end_header_id|>\n\n".to_string(),
            end_of_content: "<|eot_id|>\n".to_string(),
            stops: vec!["<|eot_id|>".to_string()],
        }
    }

    pub fn hermes_2_pro_llama3_prompt_template() -> PromptTemplate {
        PromptTemplate {
            header_prefix: "<|im_start|>".to_string(),
            header_suffix: "\n".to_string(),
            end_of_content: "<|im_end|>\n".to_string(),
            stops: vec!["<|im_end|>".to_string()],
        }
    }

    pub fn qwen_prompt_template() -> PromptTemplate {
        PromptTemplate {
            header_prefix: "<|im_start|>".to_string(),
            header_suffix: "\n".to_string(),
            end_of_content: "<|im_end|>\n".to_string(),
            stops: vec!["<|im_end|>".to_string()],
        }
    }

    pub fn gemma2_prompt_template() -> PromptTemplate {
        PromptTemplate {
            header_prefix: "<start_of_turn>".to_string(),
            header_suffix: "\n".to_string(),
            end_of_content: "<end_of_turn>\n".to_string(),
            stops: vec!["<end_of_turn>".to_string()],
        }
    }
}

fn main() {
    let cli = Args::parse();

    let prompt = std::fs::read_to_string(&cli.prompt_path).unwrap();
    let mut prompt: HashMap<String, Vec<Content>> = toml::from_str(&prompt).unwrap();
    let prompts = prompt.remove("content").unwrap();
    let mut prompts: Vec<_> = prompts
        .into_iter()
        .map(|content| Arc::new(content))
        .collect();

    let model_params = LlamaModelParams::default().with_n_gpu_layers(64);

    let template = match cli.model_type {
        ModelType::Llama3 => prompt_template::llama3_prompt_template(),
        ModelType::Hermes2ProLlama3 => prompt_template::hermes_2_pro_llama3_prompt_template(),
        ModelType::Gemma2 => prompt_template::gemma2_prompt_template(),
        ModelType::Qwen => prompt_template::qwen_prompt_template(),
    };

    let llm = llm::LlmModel::new(cli.model_path, model_params, template).unwrap();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(1024 * 2));
    let mut ctx = llm::LlamaCtx::new(llm.clone(), ctx_params).unwrap();

    loop {
        println!("{:#?}", prompts);
        println!("You:");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        prompts.push(Arc::new(Content {
            role: llm::Role::User,
            message: input.trim().to_string(),
        }));

        let mut stream = ctx
            .chat(ChatRequest {
                prompts: prompts.clone(),
                simple_option: llm::SimpleOption::None,
            })
            .unwrap();

        println!("Bot:");
        for token in &mut stream {
            print!("{}", token);
            std::io::stdout().flush().unwrap();
        }
        let full: String = stream.into();
        println!();
        println!("Full:{}", full);

        prompts.push(Arc::new(Content {
            role: llm::Role::Assistant,
            message: full,
        }));
    }
}
