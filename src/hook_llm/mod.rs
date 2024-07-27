use crate::llm::{ChatRequest, LlamaCtx};

pub trait Callback {
    fn get_input(&mut self) -> anyhow::Result<Option<ChatRequest>>;
    fn token_callback(&mut self, token: Token) -> anyhow::Result<()>;
}

pub struct HookLlama<Hook: Callback> {
    pub llm: LlamaCtx,
    pub hook: Hook,
}

#[derive(Debug, Clone)]
pub enum Token {
    Start,
    Chunk(String),
    End(String),
}

fn chat<C: Callback>(ctx: &mut LlamaCtx, callback: &mut C) -> anyhow::Result<()> {
    loop {
        let message = callback.get_input()?;
        let content = if let Some(content) = message {
            content
        } else {
            return Ok(());
        };

        callback.token_callback(Token::Start)?;

        let mut chat = ctx.chat(content)?;

        while let Some(t) = chat.next() {
            callback.token_callback(Token::Chunk(t))?;
        }
        callback.token_callback(Token::End(chat.into()))?;
    }
}

impl<Hook: Callback> HookLlama<Hook> {
    pub fn new(ctx: LlamaCtx, hook: Hook) -> Self {
        Self { llm: ctx, hook }
    }

    pub fn chat(&mut self) -> anyhow::Result<()> {
        chat(&mut self.llm, &mut self.hook)
    }
}
