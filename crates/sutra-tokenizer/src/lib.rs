/// Tokenization library for SutraWorks
/// 
/// Supports multiple tokenization algorithms:
/// - BPE (Byte Pair Encoding) - Used by GPT-2, GPT-3
/// - WordPiece - Used by BERT
/// - Unigram - Used by SentencePiece/mT5
/// 
/// Features:
/// - Fast tokenization with caching
/// - Vocabulary management
/// - Special token handling
/// - Encoding/decoding with offsets

pub mod error;
pub mod vocab;
pub mod bpe;
pub mod wordpiece;
pub mod unigram;
pub mod normalizer;
pub mod pretokenizer;
pub mod tokenizer;

pub use error::{TokenizerError, Result};
pub use vocab::{Vocab, VocabBuilder};
pub use bpe::{BpeTokenizer, BpeConfig};
pub use wordpiece::{WordPieceTokenizer, WordPieceConfig};
pub use unigram::{UnigramTokenizer, UnigramConfig};
pub use tokenizer::{Tokenizer, TokenizerConfig, Encoding};

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::{Tokenizer, TokenizerConfig, Encoding};
    pub use crate::{BpeTokenizer, BpeConfig};
    pub use crate::{WordPieceTokenizer, WordPieceConfig};
    pub use crate::{UnigramTokenizer, UnigramConfig};
    pub use crate::{Vocab, VocabBuilder};
    pub use crate::{TokenizerError, Result};
}
