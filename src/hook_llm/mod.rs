use crate::llm::{Content, LlamaCtx, LlamaModelContext, LlamaModelFullPromptContext};

pub trait IOHook {
    fn get_input(&mut self) -> anyhow::Result<Option<Content>>;
    fn token_callback(&mut self, token: Token) -> anyhow::Result<()>;
    fn parse_input(&mut self, content: &mut Content);
}

pub struct HookLlama<Hook: IOHook> {
    pub llm: LlamaCtx,
    pub hook: Hook,
}

#[derive(Debug, Clone)]
pub enum Token {
    Start,
    Chunk(String),
    End(String),
}

fn full_chat<Hook: IOHook>(
    ctx: &mut LlamaModelFullPromptContext,
    hook: &mut Hook,
) -> anyhow::Result<()> {
    loop {
        let message = hook.get_input()?;
        let mut content = if let Some(content) = message {
            content
        } else {
            return Ok(());
        };

        hook.parse_input(&mut content);

        let mut chat = ctx.chat(content)?;

        hook.token_callback(Token::Start)?;
        while let Some(t) = chat.next() {
            hook.token_callback(Token::Chunk(t))?;
        }
        hook.token_callback(Token::End(chat.into()))?;
    }
}

fn chat<Hook: IOHook>(ctx: &mut LlamaModelContext, hook: &mut Hook) -> anyhow::Result<()> {
    loop {
        let message = hook.get_input()?;
        let mut content = if let Some(content) = message {
            content
        } else {
            return Ok(());
        };

        hook.parse_input(&mut content);

        let mut chat = ctx.chat(content)?;

        hook.token_callback(Token::Start)?;
        while let Some(t) = chat.next() {
            hook.token_callback(Token::Chunk(t))?;
        }
        hook.token_callback(Token::End(chat.into()))?;
    }
}

impl<Hook: IOHook> HookLlama<Hook> {
    pub fn new(mut ctx: LlamaCtx, mut hook: Hook) -> Self {
        let prompts = match &mut ctx {
            LlamaCtx::Full(ctx) => &mut ctx.prompts,
            LlamaCtx::Step(ctx) => &mut ctx.system_prompt,
        };

        for prompt in prompts.iter_mut() {
            hook.parse_input(prompt);
        }

        Self { llm: ctx, hook }
    }

    pub fn chat(&mut self) -> anyhow::Result<()> {
        match &mut self.llm {
            LlamaCtx::Full(ctx) => full_chat(ctx, &mut self.hook),
            LlamaCtx::Step(ctx) => chat(ctx, &mut self.hook),
        }
    }
}
