use mlua::Lua;

use crate::llm::{self, Content, LlamaModelContext, LlamaModelFullPromptContext};

pub enum LlamaCtx {
    Full(LlamaModelFullPromptContext),
    Continuous(LlamaModelContext),
}

impl From<LlamaModelFullPromptContext> for LlamaCtx {
    fn from(llm: LlamaModelFullPromptContext) -> Self {
        LlamaCtx::Full(llm)
    }
}

impl From<LlamaModelContext> for LlamaCtx {
    fn from(llm: LlamaModelContext) -> Self {
        LlamaCtx::Continuous(llm)
    }
}

pub struct LuaLlama<Data> {
    pub llm: LlamaCtx,
    pub lua: Lua,
    pub hook: ChatHook<Data>,
}

#[derive(Debug, Clone)]
pub enum Token {
    Start,
    Chunk(String),
    End(String),
}

pub struct ChatHook<Data> {
    pub data: Data,
    pub get_user_input: fn(&mut Data) -> anyhow::Result<Option<String>>,
    pub token_callback: fn(&mut Data, token: Token) -> anyhow::Result<()>,
    pub parse_user_input: fn(&mut Data, result: &str) -> String,
    pub parse_script_result: fn(&mut Data, result: &str) -> String,
    pub parse_script_error: fn(&mut Data, err: mlua::Error) -> String,
}

fn full_chat<Data>(
    ctx: &mut LlamaModelFullPromptContext,
    lua: &Lua,
    hook: &mut ChatHook<Data>,
) -> anyhow::Result<()> {
    let mut lua_result: Option<String> = None;

    loop {
        let contont = match lua_result.take() {
            Some(s) => {
                let message = (hook.parse_script_result)(&mut hook.data, &s);
                Content {
                    role: llm::Role::User,
                    message,
                }
            }
            None => {
                let message = (hook.get_user_input)(&mut hook.data)?;
                if let Some(s) = message {
                    let message = (hook.parse_user_input)(&mut hook.data, &s);
                    Content {
                        role: llm::Role::User,
                        message,
                    }
                } else {
                    return Ok(());
                }
            }
        };

        let mut chat = ctx.chat(contont)?;

        (hook.token_callback)(&mut hook.data, Token::Start)?;
        while let Some(t) = chat.next() {
            (hook.token_callback)(&mut hook.data, Token::Chunk(t))?;
        }
        drop(chat);

        let lua_msg = &ctx.prompts.last().unwrap().message;
        (hook.token_callback)(&mut hook.data, Token::End(lua_msg.clone()))?;

        if lua_msg.is_empty() || lua_msg.starts_with("--") {
            continue;
        }

        let s = lua.load(lua_msg).eval::<Option<String>>();

        let r = match s {
            Ok(Some(s)) => Some(s),
            Ok(None) => None,
            Err(e) => Some((hook.parse_script_error)(&mut hook.data, e)),
        };
        lua_result = r;
    }
}

fn chat<Data>(
    ctx: &mut LlamaModelContext,
    lua: &Lua,
    hook: &mut ChatHook<Data>,
) -> anyhow::Result<()> {
    let mut lua_result: Option<String> = None;

    loop {
        let contont = match lua_result.take() {
            Some(s) => {
                let message = (hook.parse_script_result)(&mut hook.data, &s);
                Content {
                    role: llm::Role::User,
                    message,
                }
            }
            None => {
                let message = (hook.get_user_input)(&mut hook.data)?;
                if let Some(s) = message {
                    let message = (hook.parse_user_input)(&mut hook.data, &s);

                    Content {
                        role: llm::Role::User,
                        message,
                    }
                } else {
                    return Ok(());
                }
            }
        };

        let mut chat = ctx.chat(contont)?;

        let mut lua_msg = String::with_capacity(128);

        (hook.token_callback)(&mut hook.data, Token::Start)?;
        while let Some(t) = chat.next() {
            lua_msg += &t;
            (hook.token_callback)(&mut hook.data, Token::Chunk(t))?;
        }
        (hook.token_callback)(&mut hook.data, Token::End(lua_msg.clone()))?;

        if lua_msg.is_empty() || lua_msg.starts_with("--") {
            continue;
        }

        let s = lua.load(lua_msg).eval::<Option<String>>();

        let r = match s {
            Ok(Some(s)) => Some(s),
            Ok(None) => None,
            Err(e) => Some((hook.parse_script_error)(&mut hook.data, e)),
        };
        lua_result = r;
    }
}

impl<Data> LuaLlama<Data> {
    pub fn new(mut llm: LlamaCtx, lua: Lua, mut hook: ChatHook<Data>) -> Self {
        let prompts = match &mut llm {
            LlamaCtx::Full(ctx) => &mut ctx.prompts,
            LlamaCtx::Continuous(ctx) => &mut ctx.system_prompt,
        };

        for prompt in prompts.iter_mut() {
            match prompt.role {
                llm::Role::User => {
                    prompt.message = (hook.parse_user_input)(&mut hook.data, &prompt.message);
                }
                llm::Role::Tool => {
                    prompt.message = (hook.parse_script_result)(&mut hook.data, &prompt.message);
                }
                _ => (),
            }
        }

        Self { llm, lua, hook }
    }

    pub fn chat(&mut self) -> anyhow::Result<()> {
        match &mut self.llm {
            LlamaCtx::Full(ctx) => full_chat(ctx, &self.lua, &mut self.hook),
            LlamaCtx::Continuous(ctx) => chat(ctx, &self.lua, &mut self.hook),
        }
    }
}
