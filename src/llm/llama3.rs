pub fn llama3_prompt_template() -> super::PromptTemplate {
    super::PromptTemplate {
        header_prefix: "<|start_header_id|>".to_string(),
        header_suffix: "<|end_header_id|>\n\n".to_string(),
        end_of_content: "<|eot_id|>\n".to_string(),
        stops: vec!["<|eot_id|>".to_string()],
    }
}

pub fn hermes_2_pro_llama3_prompt_template() -> super::PromptTemplate {
    super::PromptTemplate {
        header_prefix: "<|im_start|>".to_string(),
        header_suffix: "\n".to_string(),
        end_of_content: "<|im_end|>\n".to_string(),
        stops: vec!["<|im_end|>".to_string()],
    }
}
