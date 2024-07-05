use std::{collections::HashMap, io::Write, num::NonZeroU32, sync::Arc};

use clap::{Parser, ValueEnum};
use llama_cpp_2::{context::params::LlamaContextParams, model::params::LlamaModelParams};
use llm::Content;

mod llm;
mod script_runtime;

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
}

#[derive(Debug, Clone, ValueEnum)]
enum ModelType {
    Llama3,
    Hermes2ProLlama3,
    Gemma2,
    Qwen,
}

fn chat(sys_prompt: Vec<Content>, llm: Arc<llm::LlmModel>) {
    let lua = script_runtime::lua_env().unwrap();

    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(1024 * 2));

    let mut ctx = llm::LlamaModelContext::new(llm.clone(), ctx_params).unwrap();
    ctx.system_prompt(sys_prompt).unwrap();

    let input = std::io::stdin();
    let mut lua_result = None;

    loop {
        let contont = match lua_result.take() {
            Some(s) => {
                println!("Tool:");
                let message = format!("{{ \"role\":\"tool\",\"message\":{s}}}");
                println!("{message}");
                Content {
                    role: llm::Role::User,
                    message,
                }
            }
            None => {
                println!("User:");
                let mut line = String::with_capacity(128);
                input.read_line(&mut line).unwrap();
                Content {
                    role: llm::Role::User,
                    message: serde_json::json!({
                        "role":"user",
                        "message":line
                    })
                    .to_string(),
                }
            }
        };

        let mut chat = ctx.chat(contont).unwrap();

        println!("AI:");
        let mut lua_msg = String::with_capacity(128);
        while let Some(t) = chat.next() {
            print!("{}", t);
            std::io::stdout().flush().unwrap();
            lua_msg.push_str(&t);
        }
        println!();

        if lua_msg.is_empty() || lua_msg.starts_with("--") {
            continue;
        }

        let s = lua.load(lua_msg).eval::<Option<String>>();

        let r = match s {
            Ok(Some(s)) => Some(s),
            Ok(None) => None,
            Err(e) => Some(
                serde_json::json!({
                    "role":"tool",
                    "message":{
                        "status":"error",
                        "error":e.to_string()
                    }
                })
                .to_string(),
            ),
        };
        lua_result = r;
    }
}

fn main() {
    let cli = Args::parse();

    let prompt = std::fs::read_to_string(&cli.prompt_path).unwrap();
    let mut prompt: HashMap<String, Vec<Content>> = toml::from_str(&prompt).unwrap();
    let sys_prompt = prompt.remove("content").unwrap();

    let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);

    let template = match cli.model_type {
        ModelType::Llama3 => llm::llama3::llama3_prompt_template(),
        ModelType::Hermes2ProLlama3 => llm::llama3::hermes_2_pro_llama3_prompt_template(),
        ModelType::Gemma2 => llm::gemma::gemma2_prompt_template(),
        ModelType::Qwen => llm::qwen::qwen_prompt_template(),
    };

    let llm = llm::LlmModel::new(cli.model_path, model_params, template).unwrap();
    chat(sys_prompt, llm);
}
