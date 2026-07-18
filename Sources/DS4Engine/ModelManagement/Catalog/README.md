# ModelManagement/Catalog

Contains the typed cross-family catalog used by the GUI. The three complete
DeepSeek V4 Flash variants and the single-GGUF Pro Q2 are downloadable and
selectable. The PRO Q4 package and the three monolithic GLM 5.2 GGUFs are
visible and downloadable, but not startable by the current runtime.

Each `ModelCatalogEntry` groups one or more `ModelTarget`s: a complete model
has a single main artifact, while PRO Q4 is a package of two shards. MTP is a
separate accessory and does not appear in the main model catalog.

The remote name, Hugging Face source/revision, identifier, exact available
size and SHA-256 digest are centralized here: the GUI must not duplicate file
names or infer support from the filename.

## Current matrix

| ID | Edition | Shape | Availability |
|---|---|---|---|
| `q2-imatrix` | Flash | one Q2/IQ2XXS GGUF | `runnable` |
| `q2-q4-imatrix` | Flash | one mixed Q2/Q4 GGUF | `runnable` |
| `q4-imatrix` | Flash | one Q4 GGUF | `runnable` |
| `pro-q2-imatrix` | Pro | one Q2 GGUF | `runnable` |
| `pro-q4-split` | Pro | two Q4 shards | `downloadOnly` |
| `glm-5.2-iq2-xxs` | GLM 5.2 | one IQ2_XXS GGUF | `downloadOnly` |
| `glm-5.2-q2-k` | GLM 5.2 | one Q2_K GGUF | `downloadOnly` |
| `glm-5.2-q4-k` | GLM 5.2 | one Q4_K GGUF | `downloadOnly` |

`ModelCatalogEntry.isSelectable` requires a `runnable` runtime, a single
artifact and the `mainModel` role. This rule prevents a split package or an
accessory from becoming a local model. `DeepSeekV4AccessoryCatalog.mtp`
remains separate and is not iterated by `ModelCatalogRegistry.entries`.

The availability of the single-file profiles derives from
`DeepSeekV4BackendDefinition.locallyRunnableVariants`; the catalog does not
keep a second Pro flag. When extending the catalog, pin filename and SHA-256
from the remote source, keep the already-persisted IDs, and enable a profile
only after an end-to-end validation of loader, tokenizer, decoder and shape.

`GLM52ModelCatalog` uses the `antirez/glm-5.2-gguf` repository pinned at
revision `2638b3b878f5c6cc3ae7334b8dbea1275025f52e`. The umbrella registry
`ModelCatalogRegistry` concatenates the family-specific catalogs; IDs and
filenames must be globally unique.
