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

pub struct LlmPromptTemplate {
    pub encode_string: fn(content: &[Content]) -> String,
    pub is_end_of_header: fn(token: &str) -> bool,
    // for gemma2
    pub post_handle: fn(token: String) -> Option<String>,
}

impl LlmPromptTemplate {
    pub fn identity(token: String) -> Option<String> {
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

pub struct LlamaModelChatStream<'a> {
    start_out: bool,
    llama_ctx: &'a mut LlamaModelContext,
    is_error: bool,
}

impl<'a> Iterator for LlamaModelChatStream<'a> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_error {
            return None;
        }

        loop {
            let s = self.llama_ctx.take_a_token();

            if let Err(e) = s {
                self.is_error = true;
                return Some(e.to_string());
            }

            let s = s.unwrap()?;

            let post_handle = self.llama_ctx.model.prompt_template.post_handle;
            if self.start_out {
                break post_handle(s);
            } else {
                let is_end_of_header = self.llama_ctx.model.prompt_template.is_end_of_header;
                if is_end_of_header(&s) {
                    self.start_out = true;
                }
            }
        }
    }
}

impl LlamaModelContext {
    pub fn new(
        model: Arc<LlmModel>,
        ctx_params: LlamaContextParams,
        system_prompt: Option<Vec<Content>>,
    ) -> anyhow::Result<Self> {
        let ctx = model.model.new_context(&model.backend, ctx_params)?;
        let n_tokens = ctx.n_ctx();
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

        if self.first_chat {
            let tokens = self
                .model
                .model
                .str_to_token(&encode_string(&self.system_prompt), model::AddBos::Always)?;
            for token in tokens.into_iter() {
                self.batch.add(token, self.n_cur, &[0], false)?;
                self.n_cur += 1;
            }
            self.first_chat = false;
        }

        let tokens = self
            .model
            .model
            .str_to_token(&encode_string(&[message]), model::AddBos::Never)?;

        // self.batch.clear();
        let last_index = (tokens.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens.into_iter()) {
            let is_last = i == last_index;
            self.batch.add(token, self.n_cur, &[0], is_last)?;
            self.n_cur += 1;
        }
        Ok(())
    }

    pub fn chat(&mut self, message: Content) -> anyhow::Result<LlamaModelChatStream> {
        self.user_message(message)?;
        let is_end_of_header = self.model.prompt_template.is_end_of_header;
        self.decoder = encoding_rs::UTF_8.new_decoder();

        let start_out = is_end_of_header("");
        Ok(LlamaModelChatStream {
            start_out,
            llama_ctx: self,
            is_error: false,
        })
    }

    fn take_a_token(&mut self) -> anyhow::Result<Option<String>> {
        self.ctx.decode(&mut self.batch)?;

        let candidates = self.ctx.candidates_ith(self.batch.n_tokens() - 1);
        let mut candidates_p = LlamaTokenDataArray::from_iter(candidates, false);

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
        let n_tokens = ctx.n_ctx();
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

    pub fn chat(&mut self, message: Content) -> anyhow::Result<LlamaModelFullPromptChatStream> {
        self.add_message(message);
        self.decoder = encoding_rs::UTF_8.new_decoder();
        let is_end_of_header = self.model.prompt_template.is_end_of_header;
        let start_out = is_end_of_header("");

        self.reset_batch_with_prompt()?;

        Ok(LlamaModelFullPromptChatStream {
            start_out,
            llama_ctx: self,
            content_buff: String::new(),
            is_error: false,
        })
    }

    fn reset_batch_with_prompt(&mut self) -> anyhow::Result<()> {
        self.ctx.clear_kv_cache();
        self.batch.clear();
        let encode_string = self.model.prompt_template.encode_string;

        let tokens = self
            .model
            .model
            .str_to_token(&encode_string(&self.prompts), model::AddBos::Always)?;

        self.batch.add_sequence(&tokens, 0, false)?;
        self.n_cur = tokens.len();
        Ok(())
    }

    fn take_a_token(&mut self) -> anyhow::Result<Option<String>> {
        self.ctx.decode(&mut self.batch)?;

        let candidates = self.ctx.candidates_ith(self.batch.n_tokens() - 1);
        let mut candidates_p = LlamaTokenDataArray::from_iter(candidates, false);

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

pub struct LlamaModelFullPromptChatStream<'a> {
    start_out: bool,
    llama_ctx: &'a mut LlamaModelFullPromptContext,
    content_buff: String,
    is_error: bool,
}

impl<'a> Iterator for LlamaModelFullPromptChatStream<'a> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        fn next_(stream: &mut LlamaModelFullPromptChatStream<'_>) -> Option<String> {
            if stream.is_error {
                return None;
            }

            loop {
                let s = stream.llama_ctx.take_a_token();
                if let Err(e) = s {
                    stream.is_error = true;
                    return Some(e.to_string());
                }
                let s = s.unwrap()?;
                let post_handle = stream.llama_ctx.model.prompt_template.post_handle;
                if stream.start_out {
                    break post_handle(s);
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
            self.content_buff.push_str(s);
        } else {
            let mut message = String::new();
            std::mem::swap(&mut message, &mut self.content_buff);
            self.llama_ctx.add_message(Content {
                role: Role::Assistant,
                message,
            })
        }
        r
    }
}
