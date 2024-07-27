use crate::llm::LlmPromptTemplate;

pub fn qwen_prompt_template() -> super::LlmPromptTemplate {
    fn encode_string(content: &[super::Content]) -> String {
        let mut result = String::new();
        for c in content.iter() {
            result.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                c.role, c.message
            ));
        }
        println!("Encoded:\n {}", result);
        result
    }

    fn is_end_of_header(token: &str) -> bool {
        token.ends_with("\n")
    }

    fn post_handle_content(content: &mut String) {
        if content.ends_with("<|im_end|>") {
            let bs = unsafe { content.as_mut_vec() };
            let len = bs.len();
            bs.truncate(len - "<|im_end|>".len());
        }
    }

    super::LlmPromptTemplate {
        encode_string,
        is_end_of_header,
        post_handle: LlmPromptTemplate::identity,
        post_handle_content,
    }
}
