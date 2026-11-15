# Flickr30k Wolof captions

`flickr30k_wolof.jsonl.gz`: one JSON object per Flickr30k image, 31,712 lines, sorted by split then `image_id`.

```json
{"image_id": "1000092795.jpg", "split": "train", "captions_en": ["...", "...", "...", "...", "..."], "captions_wo": ["...", "...", "...", "...", "..."]}
```

| split | images | caption pairs |
|---|---:|---:|
| train | 29,703 | 148,515 |
| validation | 1,014 | 5,070 |
| test | 995 | 4,975 |

- `captions_en`: the five original Flickr30k English captions.
- `captions_wo`: their Wolof versions, produced by machine translation (one request per image, five captions together) guided by the
  terminology and rules of [`glossary/wolof_captions.txt`](../glossary/wolof_captions.txt). They were not post-edited by native speakers:
  treat them as silver references.
- sha256: `0c480dda555c69be152c9f7ec61749996966dec36f63ece9d577267c90c9e8d7`

**Images are not redistributed.** Download Flickr30k from its official source (Flickr terms of use, non-commercial research only) and
point `image2wolof build --images` / `image2wolof cache --images` to the folder of JPEG files named by `image_id`.

If you use this data, cite Flickr30k:

> Peter Young, Alice Lai, Micah Hodosh, Julia Hockenmaier. *From image descriptions to visual denotations: New similarity metrics for
> semantic inference over event descriptions.* Transactions of the Association for Computational Linguistics, 2:67–78, 2014.
