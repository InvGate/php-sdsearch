//! document and field model for indexing.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FieldKind {
    /// tokenized + indexed + stored text
    Text,
    /// not tokenized: one term = the value as-is (`_key` fields)
    Keyword,
    /// stored only, not indexed
    Stored,
}

#[derive(Clone)]
pub struct Field {
    pub name: String,
    pub value: String,
    pub kind: FieldKind,
}

#[derive(Default)]
pub struct Document {
    fields: Vec<Field>,
}

impl Document {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, name: &str, value: &str, kind: FieldKind) {
        self.fields.push(Field {
            name: name.to_string(),
            value: value.to_string(),
            kind,
        });
    }

    pub fn fields(&self) -> &[Field] {
        &self.fields
    }
}
