# Format Spec YAML Guide

This directory contains YAML specs that describe how `label_master` should
recognize and read annotation datasets.

The main goal of these specs is to move dataset-specific field names and token
positions out of Python and into text files.

## Where Specs Live

Built-in specs live in:

- `src/label_master/format_specs/builtins/*.yaml`

Custom specs are searched in:

- `~/.label_master/formats/*.yaml`
- `<dataset_root>/format_specs/*.yaml`
- `<dataset_root>/.label_master/formats/*.yaml`

`dataset_root` means the dataset directory you open in the CLI or GUI.

## Top-Level Fields

Every format spec starts with:

```yaml
format_id: video_bbox
display_name: Video BBox
description: Video or frame-sequence tracking text annotations
parser:
  ...
```

- `format_id`: Stable internal ID for the format.
- `display_name`: Friendly label shown to people.
- `description`: Short explanation.
- `parser`: The actual parsing rules.

## Parser Kinds

The value of `parser.kind` tells `label_master` what parser family to use.

Current parser kinds:

- `json_object_dataset`
- `xml_annotation_dataset`
- `csv_bracket_bbox_dataset`
- `tokenized_image_labels`
- `tokenized_video`

Built-in specs use several of these already. Custom user-defined specs are
currently loaded only for `tokenized_video`.

## Glob Patterns

Fields such as `annotation_globs`, `label_globs`, and `csv_globs` use Python
`Path.glob(...)` style patterns.

Common examples:

- `annotations/*.txt`
  Means: all `.txt` files directly inside `annotations/`

- `annotations/**/*.txt`
  Means: all `.txt` files anywhere under `annotations/`, recursively

- `labels/**/*.txt`
  Means: all label text files anywhere under `labels/`

- `**/*.csv`
  Means: all `.csv` files anywhere in the dataset tree

Important note:

- `annotations/**/*.txt` is the usual recursive pattern
- `annotations/**.*.txt` is usually not what you want

Why? Because `**/` is the recursive directory pattern. Without the slash, you
are just matching a filename pattern, not “any directory depth”.

## 1-Based Field Positions

Token-position fields in the YAML are 1-based, not 0-based.

Example:

```yaml
class_id_field: 1
x_center_field: 2
```

That means:

- token 1 is the class ID
- token 2 is the x center

This matches the way people usually describe columns in text files.

## Image Coordinate Convention

Unless a format explicitly says otherwise, `x` and `y` values use regular image
coordinates:

- origin is the top-left corner of the image
- positive `x` moves right
- positive `y` moves down

So `xmin`/`ymin` mean the top-left corner of a box.

If a format uses `normalized_coordinates: true`, only the units change from
pixels to normalized values; the origin and axis directions stay the same.

## `json_object_dataset`

Used for JSON datasets like COCO.

Example:

```yaml
parser:
  kind: json_object_dataset
  annotations_file: annotations.json
  images_key: images
  annotations_key: annotations
  categories_key: categories
  image_fields:
    id: id
    file_name: file_name
    width: width
    height: height
  category_fields:
    id: id
    name: name
    supercategory: supercategory
  annotation_fields:
    id: id
    image_id: image_id
    class_id: category_id
    bbox: bbox
    iscrowd: iscrowd
  bbox_fields:
    xmin: 1
    ymin: 2
    width: 3
    height: 4
```

Meaning:

- `annotations_file`: JSON file to open
- `images_key`: top-level array holding image records
- `annotations_key`: top-level array holding annotations
- `categories_key`: top-level array holding classes
- `image_fields`: which keys inside each image record mean ID, file name, width, height
- `annotation_fields`: which keys inside each annotation record mean annotation ID, image ID, class ID, bbox
- `bbox_fields`: 1-based positions inside the bbox list for `xmin`, `ymin`, `width`, `height`

For COCO-style bboxes, that normally means:

```text
[xmin, ymin, width, height]
```

## `xml_annotation_dataset`

Used for XML datasets like Pascal VOC.

Example:

```yaml
parser:
  kind: xml_annotation_dataset
  annotation_globs:
    - "**/*.xml"
  root_tag: annotation
  filename_field: filename
  path_field: path
  size_width_field: size/width
  size_height_field: size/height
  object_tag: object
  object_name_field: name
  bbox_tag: bndbox
  bbox_fields:
    xmin: xmin
    ymin: ymin
    xmax: xmax
    ymax: ymax
```

Meaning:

- `annotation_globs`: where to find annotation XML files
- `root_tag`: expected root element
- `filename_field`: XML path to the image filename
- `path_field`: optional XML path to the full image path
- `size_width_field`, `size_height_field`: XML paths to image size
- `object_tag`: repeated tag containing one object each
- `object_name_field`: class-name field inside each object
- `bbox_tag`: enclosing bbox tag
- `bbox_fields`: names of the bbox children

Paths like `size/width` mean “find `<size><width>...</width></size>`”.

## `csv_bracket_bbox_dataset`

Used for CSV layouts like the Kitware example.

Example:

```yaml
parser:
  kind: csv_bracket_bbox_dataset
  csv_globs:
    - "**/*.csv"
  image_field_aliases:
    - imagefilename
    - image_filename
  bbox_column_class_map:
    airplane_bbox: airplane
    bird_bbox: bird
    drone_bbox: drone
    helicopter_bbox: helicopter
  bbox_fields:
    xmin: 1
    ymin: 2
    width: 3
    height: 4
  bbox_enclosure: "[]"
  box_separator: ";"
```

Meaning:

- `csv_globs`: where to find CSV files
- `image_field_aliases`: acceptable names for the image-path column
- `bbox_column_class_map`: maps each bbox column name to the normalized class name
- `bbox_fields`: 1-based positions inside each bracketed bbox for `xmin`, `ymin`, `width`, `height`
- `bbox_enclosure: "[]"`: each bbox cell is wrapped by `[` and `]`
- `box_separator: ";"`: multiple boxes in one cell are separated by `;`

Example cell:

```text
[65 25 30 27;197 87 23 22]
```

That means two boxes:

- `65 25 30 27`
- `197 87 23 22`

Each box is interpreted as `xmin ymin width height`.

## `tokenized_image_labels`

Used for line-oriented image datasets like YOLO.

Example:

```yaml
parser:
  kind: tokenized_image_labels
  label_globs:
    - "labels/**/*.txt"
    - "**/labels/**/*.txt"
  classes_file_name: classes.txt
  image_sizes_file_name: image_sizes.json
  image_extensions:
    - .jpg
    - .png
  path_rewrites:
    - from: /labels/
      to: /images/
  row_format:
    kind: single_object
    delimiter: whitespace
    class_id_field: 1
    x_center_field: 2
    y_center_field: 3
    width_field: 4
    height_field: 5
    normalized_coordinates: true
```

Meaning:

- `label_globs`: where to find label files
- `classes_file_name`: optional class-name file
- `image_sizes_file_name`: optional cached image-size file
- `image_extensions`: file extensions to try when resolving paired images
- `path_rewrites`: how to turn a label path into an image path guess
- `row_format.kind: single_object`: one object per line
- `delimiter`: whitespace or comma
- `*_field`: token positions for that row
- `normalized_coordinates: true`: bbox values are normalized, not pixels

About `path_rewrites`:

```yaml
- from: /labels/
  to: /images/
```

This is one-way. It means:

- start from a label file path
- replace `/labels/` with `/images/`
- try image extensions like `.jpg` or `.png`

It does not mean “rewrite both directions”.

## `tokenized_video`

Used for text annotation files where each row describes one frame and some
number of objects in that frame.

Example:

```yaml
parser:
  kind: tokenized_video
  annotation_globs:
    - annotations/*.txt
    - annotations/**/*.txt
  video_roots:
    - videos
    - data/videos
  row_format:
    kind: count_prefixed_objects
    delimiter: whitespace
    frame_index_field: 1
    object_count_field: 2
    frame_index_base: 0
    object_group_size: 5
    object_fields:
      xmin: 1
      ymin: 2
      width: 3
      height: 4
      class_name: 5
```

### `row_format.kind: count_prefixed_objects`

This means each row starts with:

1. a frame index
2. an object count

Then the row contains that many repeated object groups.

Example row:

```text
0 2 65 25 30 27 bird 197 87 23 22 bird
```

Interpretation:

- frame index = `0`
- object count = `2`
- each object group has 5 tokens:
  - token 1 in the group = `xmin`
  - token 2 in the group = `ymin`
  - token 3 in the group = `width`
  - token 4 in the group = `height`
  - token 5 in the group = `class_name`

So the two objects are:

- `(65, 25, 30, 27, bird)`
- `(197, 87, 23, 22, bird)`

### `frame_index_base`

This tells the parser whether the source rows count frames from 0 or 1.

- `frame_index_base: 0`
  Source row `0` means internal frame 0

- `frame_index_base: 1`
  Source row `1` means internal frame 0

### `object_group_size`

How many tokens belong to one object.

For example:

```yaml
object_group_size: 6
object_fields:
  xmin: 1
  ymin: 2
  width: 3
  height: 4
  class_id: 5
```

would mean token 6 exists too, but is currently ignored unless you map it to a
field.

### `class_name` vs `class_id`

Inside `object_fields`, at least one of these must exist:

- `class_name`
- `class_id`

If `class_name` is present, the class name is taken directly from the row.

If only `class_id` is present, `label_master` can still create internal classes,
but their fallback names will look like `class_<id>`.

## `score_boost`

Some parser kinds allow:

```yaml
score_boost: 0.05
```

This slightly nudges format inference toward that spec.

Use it sparingly. It is only meant for tie-breaking or light preference, not for
forcing a wrong match.

## How To Write a New Custom YAML

Today, the supported user-defined custom format path is `tokenized_video`.

Start from a small spec like this:

```yaml
format_id: my_video_format
display_name: My Video Format
description: Example custom tracking format
parser:
  kind: tokenized_video
  annotation_globs:
    - annotations/**/*.txt
  video_roots:
    - videos
  row_format:
    kind: count_prefixed_objects
    delimiter: whitespace
    frame_index_field: 1
    object_count_field: 2
    frame_index_base: 0
    object_group_size: 5
    object_fields:
      xmin: 1
      ymin: 2
      width: 3
      height: 4
      class_name: 5
```

Then ask these questions:

1. Where are the annotation files?
2. Where are the videos?
3. Does each row start with frame index and object count?
4. How many tokens belong to each object?
5. Which token positions correspond to `xmin`, `ymin`, `width`, `height`, and class?
6. Are frame numbers 0-based or 1-based?
7. Are tokens separated by spaces or commas?

## Debugging Tips

- If the format is not detected, check your `annotation_globs` first.
- If files are found but loading fails, check token counts and field positions.
- If the wrong videos are matched, check `video_roots`.
- If classes look wrong, verify whether your row uses `class_name` or `class_id`.
- If you want recursive matching, prefer `**/*.ext`, not `**.*.ext`.

## Good Examples to Copy

- `builtins/video_bbox.yaml`
- `builtins/yolo.yaml`
- `builtins/coco.yaml`
- `builtins/voc.yaml`
- `builtins/kitware.yaml`

For custom authoring, `video_bbox.yaml` is the closest example because it uses
the same `tokenized_video` family that custom specs currently support.
