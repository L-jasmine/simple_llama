use std::{fmt::Display, sync::Arc};

use llama_cpp_2::{
    context::LlamaContext,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{self, LlamaModel, Special},
    token::data_array::LlamaTokenDataArray,
};

pub mod gemma;
pub mod llama3;
pub mod qwen;

pub use llama_cpp_2::context::params::LlamaContextParams;
pub use llama_cpp_2::model::params::LlamaModelParams;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Role {
    #[serde(rename = "system")]
    System,
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
    #[serde(rename = "tool")]
    Tool,
    #[serde(untagged)]
    Other(String),
}

impl Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let role = match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            Role::Other(s) => s.as_str(),
        };
        write!(f, "{role}")
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Content {
    pub role: Role,
    pub message: String,
}

pub enum ChatRequest {
    Once(Content, SimpleOption),
    Full(Vec<Content>, SimpleOption),
}

#[derive(Debug, Clone, Copy)]
pub enum SimpleOption {
    None,
    Temp(f32),
    TopP(f32, usize),
}

impl Default for SimpleOption {
    fn default() -> Self {
        Self::None
    }
}

pub struct LlmPromptTemplate {
    pub encode_string: fn(content: &[Content]) -> String,
    pub is_end_of_header: fn(token: &str) -> bool,
    pub post_handle: fn(token: String, content: &str) -> Option<String>,
    pub post_handle_content: fn(content: &mut String),
}

impl LlmPromptTemplate {
    pub fn identity(token: String, _content: &str) -> Option<String> {
        Some(token)
    }
}

#[allow(unused)]
pub struct LlmModel {
    pub model_path: String,
    pub model: LlamaModel,
    pub model_params: LlamaModelParams,
    pub backend: LlamaBackend,
    pub prompt_template: LlmPromptTemplate,
}

impl LlmModel {
    pub fn new(
        model_path: String,
        model_params: LlamaModelParams,
        prompt_template: LlmPromptTemplate,
    ) -> llama_cpp_2::Result<Arc<Self>> {
        let backend = LlamaBackend::init()?;
        let llama = LlamaModel::load_from_file(&backend, &model_path, &model_params)?;
        let model = Self {
            model_path,
            model: llama,
            model_params,
            backend,
            prompt_template,
        };

        Ok(Arc::new(model))
    }
}
pub struct LlamaModelContext {
    decoder: encoding_rs::Decoder,
    ctx: LlamaContext<'static>,
    batch: LlamaBatch,
    model: Arc<LlmModel>,
    pub system_prompt: Vec<Content>,
    first_chat: bool,
    n_cur: i32,
}

pub struct LlamaModelChatStream<'a, CTX> {
    content: String,
    start_out: bool,
    llama_ctx: &'a mut CTX,
    is_error: bool,
    simple_option: SimpleOption,
}

impl<'a, CTX> Into<String> for LlamaModelChatStream<'a, CTX> {
    fn into(self) -> String {
        self.content
    }
}

impl LlamaModelContext {
    pub fn new(
        model: Arc<LlmModel>,
        ctx_params: LlamaContextParams,
        system_prompt: Option<Vec<Content>>,
    ) -> anyhow::Result<Self> {
        let ctx = model.model.new_context(&model.backend, ctx_params)?;
        let n_tokens = ctx.n_batch();
        let ctx = unsafe { std::mem::transmute(ctx) };
        let batch = LlamaBatch::new(n_tokens as usize, 1);
        let decoder = encoding_rs::UTF_8.new_decoder();

        Ok(Self {
            decoder,
            ctx,
            model,
            batch,
            n_cur: 0,
            system_prompt: system_prompt.unwrap_or_default(),
            first_chat: true,
        })
    }

    pub fn user_message(&mut self, message: Content) -> anyhow::Result<()> {
        let encode_string = self.model.prompt_template.encode_string;

        let mut tokens = self
            .model
            .model
            .str_to_token(&encode_string(&[message]), model::AddBos::Always)?;

        if self.first_chat {
            let system_tokens = self
                .model
                .model
                .str_to_token(&encode_string(&self.system_prompt), model::AddBos::Never)?;

            tokens.extend(system_tokens);

            self.first_chat = false;
        }

        self.batch.clear();

        let n_tokens = self.ctx.n_batch();

        let last_index = (tokens.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens.into_iter()) {
            let is_last = i == last_index;
            self.batch.add(token, self.n_cur, &[0], is_last)?;
            self.n_cur += 1;

            if !is_last && self.batch.n_tokens() == n_tokens as i32 {
                self.ctx.decode(&mut self.batch)?;
                self.batch.clear();
            }
        }
        Ok(())
    }

    pub fn chat(&mut self, request: ChatRequest) -> anyhow::Result<LlamaModelChatStream<Self>> {
        match request {
            ChatRequest::Once(message, simple_option) => {
                self.user_message(message)?;
                let is_end_of_header = self.model.prompt_template.is_end_of_header;
                self.decoder = encoding_rs::UTF_8.new_decoder();

                let start_out = is_end_of_header("");
                Ok(LlamaModelChatStream {
                    content: String::with_capacity(128),
                    start_out,
                    llama_ctx: self,
                    is_error: false,
                    simple_option,
                })
            }
            ChatRequest::Full(_, _) => Err(anyhow::anyhow!("no support for full chat")),
        }
    }

    fn take_a_token(&mut self, simple_option: SimpleOption) -> anyhow::Result<Option<String>> {
        self.ctx.decode(&mut self.batch)?;

        let candidates = self.ctx.candidates_ith(self.batch.n_tokens() - 1);
        let mut candidates_p = LlamaTokenDataArray::from_iter(candidates, false);
        match simple_option {
            SimpleOption::None => {}
            SimpleOption::Temp(temperature) => candidates_p.sample_temp(None, temperature),
            SimpleOption::TopP(p, min_keep) => candidates_p.sample_top_p(None, p, min_keep),
        }

        let new_token_id = candidates_p.sample_token(&mut self.ctx);

        self.batch.clear();
        self.batch.add(new_token_id, self.n_cur, &[0], true)?;
        self.n_cur += 1;

        if new_token_id == self.model.model.token_eos() {
            return Ok(None);
        } else {
            let output_bytes = self
                .model
                .model
                .token_to_bytes(new_token_id, Special::Tokenize)?;
            let mut output_string = String::with_capacity(32);
            let _decode_result =
                self.decoder
                    .decode_to_string(&output_bytes, &mut output_string, false);

            Ok(Some(output_string))
        }
    }
}

pub struct LlamaModelFullPromptContext {
    decoder: encoding_rs::Decoder,
    ctx: LlamaContext<'static>,
    batch: LlamaBatch,
    model: Arc<LlmModel>,
    pub prompts: Vec<Content>,
    n_cur: usize,
}

impl LlamaModelFullPromptContext {
    pub fn new(
        model: Arc<LlmModel>,
        ctx_params: LlamaContextParams,
        system_prompts: Option<Vec<Content>>,
    ) -> anyhow::Result<Self> {
        let ctx = model.model.new_context(&model.backend, ctx_params)?;
        let n_tokens = ctx.n_batch();
        let ctx = unsafe { std::mem::transmute(ctx) };
        let batch = LlamaBatch::new(n_tokens as usize, 1);
        let decoder = encoding_rs::UTF_8.new_decoder();

        Ok(Self {
            decoder,
            ctx,
            model,
            batch,
            prompts: system_prompts.unwrap_or_default(),
            n_cur: 0,
        })
    }

    pub fn add_message(&mut self, message: Content) {
        self.prompts.push(message);
    }

    pub fn chat(&mut self, request: ChatRequest) -> anyhow::Result<LlamaModelChatStream<Self>> {
        match request {
            ChatRequest::Once(message, simple_option) => {
                self.add_message(message);
                self.decoder = encoding_rs::UTF_8.new_decoder();
                let is_end_of_header = self.model.prompt_template.is_end_of_header;
                let start_out = is_end_of_header("");

                self.reset_batch_with_prompt(None)?;

                Ok(LlamaModelChatStream {
                    content: String::with_capacity(128),
                    start_out,
                    llama_ctx: self,
                    is_error: false,
                    simple_option,
                })
            }
            ChatRequest::Full(prompts, simple_option) => {
                self.decoder = encoding_rs::UTF_8.new_decoder();
                let is_end_of_header = self.model.prompt_template.is_end_of_header;
                let start_out = is_end_of_header("");

                self.reset_batch_with_prompt(Some(&prompts))?;

                Ok(LlamaModelChatStream {
                    content: String::with_capacity(128),
                    start_out,
                    llama_ctx: self,
                    is_error: false,
                    simple_option,
                })
            }
        }
    }

    fn reset_batch_with_prompt(&mut self, prompts: Option<&[Content]>) -> anyhow::Result<()> {
        self.ctx.clear_kv_cache();
        self.batch.clear();
        self.n_cur = 0;

        let encode_string = self.model.prompt_template.encode_string;

        let prompts = prompts.unwrap_or(&self.prompts);

        let tokens = self
            .model
            .model
            .str_to_token(&encode_string(prompts), model::AddBos::Always)?;

        let last_index = (tokens.len() - 1) as i32;
        let n_tokens = self.ctx.n_batch();

        for (i, token) in (0_i32..).zip(tokens.into_iter()) {
            let is_last = i == last_index;

            self.batch.add(token, self.n_cur as i32, &[0], is_last)?;
            self.n_cur += 1;

            if !is_last && self.batch.n_tokens() == n_tokens as i32 {
                self.ctx.decode(&mut self.batch)?;
                self.batch.clear();
            }
        }

        Ok(())
    }

    fn take_a_token(&mut self, simple_option: SimpleOption) -> anyhow::Result<Option<String>> {
        self.ctx.decode(&mut self.batch)?;

        let candidates = self.ctx.candidates_ith(self.batch.n_tokens() - 1);
        let mut candidates_p = LlamaTokenDataArray::from_iter(candidates, false);
        match simple_option {
            SimpleOption::None => {}
            SimpleOption::Temp(temperature) => candidates_p.sample_temp(None, temperature),
            SimpleOption::TopP(p, min_keep) => candidates_p.sample_top_p(None, p, min_keep),
        }

        let new_token_id = candidates_p.sample_token(&mut self.ctx);

        self.batch.clear();
        self.batch
            .add(new_token_id, self.n_cur as i32, &[0], true)?;
        self.n_cur += 1;

        if new_token_id == self.model.model.token_eos() {
            return Ok(None);
        } else {
            let output_bytes = self
                .model
                .model
                .token_to_bytes(new_token_id, Special::Tokenize)?;
            let mut output_string = String::with_capacity(32);
            let _decode_result =
                self.decoder
                    .decode_to_string(&output_bytes, &mut output_string, false);

            Ok(Some(output_string))
        }
    }
}

pub enum LlamaCtx {
    Full(LlamaModelFullPromptContext),
    Step(LlamaModelContext),
}

impl From<LlamaModelFullPromptContext> for LlamaCtx {
    fn from(llm: LlamaModelFullPromptContext) -> Self {
        LlamaCtx::Full(llm)
    }
}

impl From<LlamaModelContext> for LlamaCtx {
    fn from(llm: LlamaModelContext) -> Self {
        LlamaCtx::Step(llm)
    }
}

impl<'a> Iterator for LlamaModelChatStream<'a, LlamaModelContext> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_error {
            return None;
        }

        let r = loop {
            let s = self.llama_ctx.take_a_token(self.simple_option);

            if let Err(e) = s {
                self.is_error = true;
                return Some(e.to_string());
            }

            let s = s.unwrap()?;

            let post_handle = self.llama_ctx.model.prompt_template.post_handle;
            if self.start_out {
                break post_handle(s, &self.content);
            } else {
                let is_end_of_header = self.llama_ctx.model.prompt_template.is_end_of_header;
                if is_end_of_header(&s) {
                    self.start_out = true;
                }
            }
        };
        if let Some(r) = &r {
            self.content.push_str(r);
        } else {
            let post_handle_content = self.llama_ctx.model.prompt_template.post_handle_content;
            post_handle_content(&mut self.content)
        }

        r
    }
}

impl<'a> Iterator for LlamaModelChatStream<'a, LlamaModelFullPromptContext> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        fn next_(
            stream: &mut LlamaModelChatStream<'_, LlamaModelFullPromptContext>,
        ) -> Option<String> {
            if stream.is_error {
                return None;
            }

            loop {
                let s = stream.llama_ctx.take_a_token(stream.simple_option);
                if let Err(e) = s {
                    stream.is_error = true;
                    return Some(e.to_string());
                }
                let s = s.unwrap()?;
                let post_handle = stream.llama_ctx.model.prompt_template.post_handle;
                if stream.start_out {
                    break post_handle(s, &stream.content);
                } else {
                    let is_end_of_header = stream.llama_ctx.model.prompt_template.is_end_of_header;
                    if is_end_of_header(&s) {
                        stream.start_out = true;
                    }
                }
            }
        }

        let r = next_(self);
        if let Some(s) = &r {
            self.content.push_str(s);
        } else {
            let post_handle_content = self.llama_ctx.model.prompt_template.post_handle_content;
            post_handle_content(&mut self.content);
            self.llama_ctx.add_message(Content {
                role: crate::llm::Role::Assistant,
                message: self.content.clone(),
            });
        }
        r
    }
}
