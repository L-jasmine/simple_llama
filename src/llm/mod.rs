use std::{fmt::Display, sync::Arc};

use llama_cpp_2::{
    context::{params::LlamaContextParams, LlamaContext},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{self, params::LlamaModelParams, LlamaModel, Special},
    token::data_array::LlamaTokenDataArray,
};

pub mod gemma;
pub mod llama3;
pub mod qwen;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    pub encode_string: fn(content: &[super::Content]) -> String,
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
    system_prompt: Vec<super::Content>,
    first_chat: bool,
    n_cur: i32,
}

pub struct LlamaModelChatStream<'a> {
    start_out: bool,
    llama_ctx: &'a mut LlamaModelContext,
}

impl<'a> Iterator for LlamaModelChatStream<'a> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let s = self.llama_ctx.take_a_token().ok()??;
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
    pub fn new(model: Arc<LlmModel>, ctx_params: LlamaContextParams) -> anyhow::Result<Self> {
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
            system_prompt: Vec::new(),
            first_chat: true,
        })
    }

    pub fn system_prompt(&mut self, prompt: Vec<super::Content>) -> anyhow::Result<()> {
        self.system_prompt = prompt;
        Ok(())
    }

    pub fn user_message(&mut self, message: super::Content) -> anyhow::Result<()> {
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

    pub fn chat(&mut self, message: super::Content) -> anyhow::Result<LlamaModelChatStream> {
        self.user_message(message)?;
        let is_end_of_header = self.model.prompt_template.is_end_of_header;
        self.decoder = encoding_rs::UTF_8.new_decoder();

        let start_out = is_end_of_header("");
        Ok(LlamaModelChatStream {
            start_out,
            llama_ctx: self,
        })
    }

    pub fn take_a_token(&mut self) -> anyhow::Result<Option<String>> {
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
