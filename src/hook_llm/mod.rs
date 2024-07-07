use crate::llm::{Content, LlamaModelContext, LlamaModelFullPromptContext};

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
        drop(chat);

        let llama_out = ctx
            .prompts
            .last()
            .map(|s| s.message.clone())
            .unwrap_or_default();

        hook.token_callback(Token::End(llama_out))?;
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

        let mut llama_out = String::with_capacity(128);

        hook.token_callback(Token::Start)?;
        while let Some(t) = chat.next() {
            llama_out += &t;
            hook.token_callback(Token::Chunk(t))?;
        }

        drop(chat);

        hook.token_callback(Token::End(llama_out))?;
    }
}

impl<Hook: IOHook> HookLlama<Hook> {
    pub fn new(mut ctx: LlamaCtx, mut hook: Hook) -> Self {
        let prompts = match &mut ctx {
            LlamaCtx::Full(ctx) => &mut ctx.prompts,
            LlamaCtx::Continuous(ctx) => &mut ctx.system_prompt,
        };

        for prompt in prompts.iter_mut() {
            hook.parse_input(prompt);
        }

        Self { llm: ctx, hook }
    }

    pub fn chat(&mut self) -> anyhow::Result<()> {
        match &mut self.llm {
            LlamaCtx::Full(ctx) => full_chat(ctx, &mut self.hook),
            LlamaCtx::Continuous(ctx) => chat(ctx, &mut self.hook),
        }
    }
}
