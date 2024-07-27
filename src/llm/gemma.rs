pub fn gemma2_prompt_template() -> super::PromptTemplate {
    super::PromptTemplate {
        header_prefix: "<start_of_turn>".to_string(),
        header_suffix: "\n".to_string(),
        end_of_content: "<end_of_turn>\n".to_string(),
        stops: vec!["<end_of_turn>".to_string()],
    }
}
