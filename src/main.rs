use clap::{Parser, ValueEnum};
use std::{collections::HashMap, io::Write, num::NonZeroU32};

use lua_llama::{
    llm::{self, Content, LlamaContextParams, LlamaModelParams},
    script_llm::{self, ChatHook, Token},
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
    use mlua::prelude::*;

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

        // set_timer(time:int,text:string) // 这个函数可以设置一个定时器，time是时间间隔(s)，func是回调函数
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

    let model_params = LlamaModelParams::default().with_n_gpu_layers(512);

    let template = match cli.model_type {
        ModelType::Llama3 => llm::llama3::llama3_prompt_template(),
        ModelType::Hermes2ProLlama3 => llm::llama3::hermes_2_pro_llama3_prompt_template(),
        ModelType::Gemma2 => llm::gemma::gemma2_prompt_template(),
        ModelType::Qwen => llm::qwen::qwen_prompt_template(),
    };

    let llm = llm::LlmModel::new(cli.model_path, model_params, template).unwrap();
    let lua = lua_env::new_lua_env().unwrap();

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

    {
        fn get_user_input(data: &mut std::io::Stdin) -> anyhow::Result<Option<String>> {
            println!("User:");
            let mut line = String::with_capacity(128);
            data.read_line(&mut line)?;
            let line = serde_json::json!({
                "role":"user",
                "message":line
            })
            .to_string();
            Ok(Some(line))
        }

        fn token_callback(_data: &mut std::io::Stdin, token: Token) -> anyhow::Result<()> {
            match token {
                Token::Start => println!("AI:"),
                Token::Chunk(t) => {
                    print!("{}", t);
                    std::io::stdout().flush().unwrap();
                }
                Token::End => println!(),
            };
            Ok(())
        }

        fn parse_script_result(_data: &mut std::io::Stdin, result: &str) -> anyhow::Result<String> {
            let message = format!("{{ \"role\":\"tool\",\"message\":{result}}}");
            println!("Tool:");
            println!("{message}");
            Ok(message)
        }

        fn parse_script_error(
            _data: &mut std::io::Stdin,
            err: mlua::Error,
        ) -> anyhow::Result<String> {
            let message = serde_json::json!({
                "role":"tool",
                "message":{
                    "status":"error",
                    "error":err.to_string()
                }
            })
            .to_string();
            println!("{message}");
            Ok(message)
        }

        let hook = ChatHook {
            data: std::io::stdin(),
            get_user_input,
            token_callback,
            parse_script_result,
            parse_script_error,
        };
        let mut script_llm = script_llm::LuaLlama {
            llm: ctx,
            lua,
            hook,
        };

        script_llm.chat().unwrap();
    };
}
