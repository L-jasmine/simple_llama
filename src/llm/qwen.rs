pub fn qwen_prompt_template() -> super::PromptTemplate {
    super::PromptTemplate {
        header_prefix: "<|im_start|>".to_string(),
        header_suffix: "\n".to_string(),
        end_of_content: "<|im_end|>\n".to_string(),
        stops: vec!["<|im_end|>".to_string()],
    }
}
