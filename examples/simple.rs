use clap::{Parser, ValueEnum};
use lua_env::new_lua_hook;
use std::{collections::HashMap, num::NonZeroU32};

use simple_llama::{
    hook_llm,
    llm::{self, Content, LlamaContextParams, LlamaModelParams},
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

mod lua_env {
    use std::io::Write;

    use mlua::prelude::*;
    use simple_llama::{
        llm::{Content, Role},
        IOHook,
    };

    pub fn new_lua_env() -> Result<Lua, LuaError> {
        let lua = Lua::new();

        let reply = lua.create_function(|_, sms_msg: String| -> LuaResult<Option<String>> {
            println!("reply: {}", sms_msg);
            Ok(None)
        })?;

        let send_sms = lua.create_function(
            |_, (number, sms_msg): (String, String)| -> LuaResult<String> {
                println!("lua: Sending SMS to {}: {}", number, sms_msg);

                let s = serde_json::json!({
                    "status":"ok"
                })
                .to_string();

                Ok(s)
            },
        )?;

        let send_msg = lua.create_function(
            |_, (room_id, message): (u64, String)| -> LuaResult<String> {
                println!("lua: Sending message to room {}: {}", room_id, message);

                let s = serde_json::json!({
                    "status":"ok"
                })
                .to_string();

                Ok(s)
            },
        )?;

        let remember =
            lua.create_function(|_, (time, text): (u64, String)| -> LuaResult<String> {
                println!("set_timer {time}: {text}");
                let s = serde_json::json!({
                    "status":"ok"
                })
                .to_string();
                Ok(s)
            })?;

        let get_weather = lua.create_function(|_, _: ()| -> LuaResult<String> {
            println!("get_weather");
            Ok("下雨".to_string())
        })?;

        lua.globals().set("reply", reply)?;
        lua.globals().set("send_sms", send_sms)?;
        lua.globals().set("send_msg", send_msg)?;
        lua.globals().set("remember", remember)?;
        lua.globals().set("get_weather", get_weather)?;

        Ok(lua)
    }

    pub struct LuaHook {
        lua: Lua,
        lua_result: Option<String>,
        stdin: std::io::Stdin,
    }

    impl IOHook for LuaHook {
        fn get_input(&mut self) -> anyhow::Result<Option<simple_llama::llm::Content>> {
            if let Some(lua_result) = self.lua_result.take() {
                println!("Lua:");
                println!("{}", lua_result);
                let c = Content {
                    role: Role::Tool,
                    message: lua_result,
                };
                Ok(Some(c))
            } else {
                println!("User:");
                let mut line = String::with_capacity(64);
                self.stdin.read_line(&mut line)?;
                let c = Content {
                    role: Role::User,
                    message: line,
                };
                Ok(Some(c))
            }
        }

        fn token_callback(&mut self, token: simple_llama::Token) -> anyhow::Result<()> {
            match token {
                simple_llama::Token::Start => println!("AI:"),
                simple_llama::Token::Chunk(t) => {
                    print!("{}", t);
                    std::io::stdout().flush().unwrap();
                }
                // simple_llama::Token::End(_) => {
                //     println!();
                // }
                simple_llama::Token::End(full_output) => {
                    println!();
                    if full_output.is_empty() || full_output.starts_with("--") {
                        return Ok(());
                    } else {
                        let s = self.lua.load(full_output).eval::<Option<String>>();

                        let r = match s {
                            Ok(Some(s)) => Some(s),
                            Ok(None) => None,
                            Err(err) => Some(
                                serde_json::json!(
                                    {
                                        "status":"error",
                                        "error":err.to_string()
                                    }
                                )
                                .to_string(),
                            ),
                        };
                        self.lua_result = r;
                    }
                }
            }
            Ok(())
        }

        fn parse_input(&mut self, content: &mut simple_llama::llm::Content) {
            match content.role {
                Role::User => {
                    content.message = serde_json::json!({
                        "role":"user",
                        "message":content.message
                    })
                    .to_string();
                }
                Role::Tool => {
                    content.role = Role::User;
                    content.message =
                        format!("{{ \"role\":\"tool\",\"message\":{}}}", content.message);
                }
                _ => {}
            }
        }
    }

    pub fn new_lua_hook() -> Result<LuaHook, LuaError> {
        let lua = new_lua_env()?;
        Ok(LuaHook {
            lua,
            lua_result: None,
            stdin: std::io::stdin(),
        })
    }

    #[test]
    fn run() {
        let lua = new_lua_env().unwrap();
        let s = lua.load("send_sms('10086','gg')").eval::<String>();
        println!("{s:?}")
    }
}

fn main() {
    let cli = Args::parse();

    let prompt = std::fs::read_to_string(&cli.prompt_path).unwrap();
    let mut prompt: HashMap<String, Vec<Content>> = toml::from_str(&prompt).unwrap();
    let sys_prompt = prompt.remove("content").unwrap();

    let model_params = LlamaModelParams::default().with_n_gpu_layers(64);

    let template = match cli.model_type {
        ModelType::Llama3 => llm::llama3::llama3_prompt_template(),
        ModelType::Hermes2ProLlama3 => llm::llama3::hermes_2_pro_llama3_prompt_template(),
        ModelType::Gemma2 => llm::gemma::gemma2_prompt_template(),
        ModelType::Qwen => llm::qwen::qwen_prompt_template(),
    };

    let llm = llm::LlmModel::new(cli.model_path, model_params, template).unwrap();

    let ctx = if cli.full_chat {
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(1024 * 2));
        llm::LlamaModelFullPromptContext::new(llm.clone(), ctx_params, Some(sys_prompt))
            .unwrap()
            .into()
    } else {
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(1024 * 2));
        llm::LlamaModelContext::new(llm.clone(), ctx_params, Some(sys_prompt))
            .unwrap()
            .into()
    };

    let mut script_llm = hook_llm::HookLlama {
        llm: ctx,
        hook: new_lua_hook().unwrap(),
    };

    script_llm.chat().unwrap();
}
