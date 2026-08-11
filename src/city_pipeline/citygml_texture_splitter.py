"""Split a textured CityGML file into Mapray upload-sized datasets."""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import quoteattr

from mapray_b3d_uploader import MaprayUploadError, build_upload_plan


CORE_NS = "http://www.opengis.net/citygml/2.0"
APP_NS = "http://www.opengis.net/citygml/appearance/2.0"
GML_NS = "http://www.opengis.net/gml"
CITY_OBJECT_MEMBER = f"{{{CORE_NS}}}cityObjectMember"
BOUNDED_BY = f"{{{GML_NS}}}boundedBy"
APPEARANCE = f"{{{APP_NS}}}Appearance"
THEME = f"{{{APP_NS}}}theme"
SURFACE_DATA_MEMBER = f"{{{APP_NS}}}surfaceDataMember"
PARAMETERIZED_TEXTURE = f"{{{APP_NS}}}ParameterizedTexture"
IMAGE_URI = f"{{{APP_NS}}}imageURI"
TARGET = f"{{{APP_NS}}}target"
GML_ID = f"{{{GML_NS}}}id"


@dataclass(frozen=True)
class TextureBinding:
    uri: str
    target_ids: frozenset[str]
    owners: frozenset[int]


@dataclass(frozen=True)
class SplitChunk:
    index: int
    city_objects: frozenset[int]
    texture_uris: frozenset[str]


@dataclass(frozen=True)
class CityGmlAnalysis:
    source: Path
    namespaces: tuple[tuple[str, str], ...]
    root_tag: str
    root_attributes: tuple[tuple[str, str], ...]
    bounded_by_xml: str
    appearance_attributes: tuple[tuple[str, str], ...]
    theme_xml: str
    city_ids: tuple[frozenset[str], ...]
    geometry_owner: dict[str, int]
    textures: tuple[TextureBinding, ...]
    building_textures: tuple[frozenset[str], ...]


def _normalized_uri(value: str) -> str:
    return value.strip().replace("\\", "/")


def _target_id(value: str) -> str:
    value = value.strip()
    return value[1:] if value.startswith("#") else value


def analyze_citygml(source: Path) -> CityGmlAnalysis:
    source = source.resolve()
    if not source.is_file():
        raise MaprayUploadError(f"CityGML file does not exist: {source}")

    namespaces: list[tuple[str, str]] = []
    namespace_seen: set[tuple[str, str]] = set()
    root_tag = ""
    root_attributes: tuple[tuple[str, str], ...] = ()
    bounded_by_xml = ""
    appearance_attributes: tuple[tuple[str, str], ...] = ()
    theme_xml = f"<app:theme>rgbTexture</app:theme>"
    city_ids: list[frozenset[str]] = []
    geometry_owner: dict[str, int] = {}
    texture_targets: dict[str, set[str]] = {}
    texture_order: list[str] = []

    context = ET.iterparse(source, events=("start", "end", "start-ns"))
    for event, value in context:
        if event == "start-ns":
            prefix, uri = value
            entry = (prefix or "", uri)
            if entry not in namespace_seen:
                namespace_seen.add(entry)
                namespaces.append(entry)
                if prefix != "xml":
                    ET.register_namespace(prefix or "", uri)
            continue

        element = value
        if event == "start":
            if not root_tag:
                root_tag = element.tag
                root_attributes = tuple(element.attrib.items())
            elif element.tag == APPEARANCE and not appearance_attributes:
                appearance_attributes = tuple(element.attrib.items())
            continue

        if element.tag == BOUNDED_BY and not bounded_by_xml:
            bounded_by_xml = ET.tostring(element, encoding="unicode")
            element.clear()
        elif element.tag == CITY_OBJECT_MEMBER:
            index = len(city_ids)
            identifiers = frozenset(
                candidate.attrib[GML_ID]
                for candidate in element.iter()
                if GML_ID in candidate.attrib
            )
            for identifier in identifiers:
                previous = geometry_owner.setdefault(identifier, index)
                if previous != index:
                    raise MaprayUploadError(f"duplicate gml:id in CityGML: {identifier}")
            city_ids.append(identifiers)
            element.clear()
        elif element.tag == THEME:
            theme_xml = ET.tostring(element, encoding="unicode")
        elif element.tag == SURFACE_DATA_MEMBER:
            image = element.find(f".//{IMAGE_URI}")
            uri = _normalized_uri(image.text or "") if image is not None else ""
            if not uri:
                raise MaprayUploadError("appearance surfaceDataMember has no imageURI")
            if uri not in texture_targets:
                texture_targets[uri] = set()
                texture_order.append(uri)
            for target in element.findall(f".//{TARGET}"):
                target_id = _target_id(target.attrib.get("uri", ""))
                if target_id:
                    texture_targets[uri].add(target_id)
            element.clear()

    if not root_tag or not bounded_by_xml:
        raise MaprayUploadError("CityGML root or boundedBy element is missing")
    if not city_ids:
        raise MaprayUploadError("CityGML has no cityObjectMember")

    unresolved: list[str] = []
    textures: list[TextureBinding] = []
    building_textures: list[set[str]] = [set() for _ in city_ids]
    for uri in texture_order:
        target_ids = texture_targets[uri]
        owners: set[int] = set()
        for target_id in target_ids:
            owner = geometry_owner.get(target_id)
            if owner is None:
                unresolved.append(target_id)
            else:
                owners.add(owner)
                building_textures[owner].add(uri)
        textures.append(
            TextureBinding(uri, frozenset(target_ids), frozenset(owners))
        )
    if unresolved:
        sample = ", ".join(sorted(set(unresolved))[:5])
        raise MaprayUploadError(
            f"appearance has {len(unresolved)} unresolved target(s): {sample}"
        )

    return CityGmlAnalysis(
        source=source,
        namespaces=tuple(namespaces),
        root_tag=root_tag,
        root_attributes=root_attributes,
        bounded_by_xml=bounded_by_xml,
        appearance_attributes=appearance_attributes,
        theme_xml=theme_xml,
        city_ids=tuple(city_ids),
        geometry_owner=geometry_owner,
        textures=tuple(textures),
        building_textures=tuple(frozenset(values) for values in building_textures),
    )


def build_split_chunks(
    analysis: CityGmlAnalysis, max_textures: int = 999
) -> tuple[SplitChunk, ...]:
    if max_textures < 1 or max_textures > 999:
        raise ValueError("max_textures must be between 1 and 999")

    chunks: list[SplitChunk] = []
    current_objects: set[int] = set()
    current_textures: set[str] = set()
    for index, textures in enumerate(analysis.building_textures):
        if len(textures) > max_textures:
            raise MaprayUploadError(
                f"cityObjectMember {index} alone references {len(textures)} textures"
            )
        combined = current_textures | set(textures)
        if current_objects and len(combined) > max_textures:
            chunks.append(
                SplitChunk(
                    index=len(chunks) + 1,
                    city_objects=frozenset(current_objects),
                    texture_uris=frozenset(current_textures),
                )
            )
            current_objects = set()
            current_textures = set()
        current_objects.add(index)
        current_textures.update(textures)

    if current_objects:
        chunks.append(
            SplitChunk(
                index=len(chunks) + 1,
                city_objects=frozenset(current_objects),
                texture_uris=frozenset(current_textures),
            )
        )
    return tuple(chunks)


def _qualified_name(name: str, namespaces: tuple[tuple[str, str], ...]) -> str:
    if not name.startswith("{"):
        return name
    uri, local = name[1:].split("}", 1)
    for prefix, namespace_uri in namespaces:
        if namespace_uri == uri:
            return f"{prefix}:{local}" if prefix else local
    raise MaprayUploadError(f"namespace is not declared for XML name: {name}")


def _start_tag(
    tag: str,
    attributes: tuple[tuple[str, str], ...],
    namespaces: tuple[tuple[str, str], ...] = (),
    declare_namespaces: bool = False,
) -> str:
    parts = [f"<{_qualified_name(tag, namespaces)}"]
    if declare_namespaces:
        for prefix, uri in namespaces:
            name = f"xmlns:{prefix}" if prefix else "xmlns"
            parts.append(f" {name}={quoteattr(uri)}")
    for name, value in attributes:
        parts.append(f" {_qualified_name(name, namespaces)}={quoteattr(value)}")
    parts.append(">")
    return "".join(parts)


def _end_tag(tag: str, namespaces: tuple[tuple[str, str], ...]) -> str:
    return f"</{_qualified_name(tag, namespaces)}>"


def _surface_for_chunk(
    element: ET.Element,
    chunk: SplitChunk,
    geometry_owner: dict[str, int],
) -> str:
    clone = ET.fromstring(ET.tostring(element, encoding="utf-8"))
    for parameterized in clone.iter(PARAMETERIZED_TEXTURE):
        for target in list(parameterized.findall(TARGET)):
            target_id = _target_id(target.attrib.get("uri", ""))
            if geometry_owner.get(target_id) not in chunk.city_objects:
                parameterized.remove(target)
    return ET.tostring(clone, encoding="unicode")


def split_citygml(
    analysis: CityGmlAnalysis,
    output_root: Path,
    max_textures: int = 999,
) -> dict[str, object]:
    output_root = output_root.resolve()
    partial_root = output_root.with_name(output_root.name + ".partial")
    if output_root.exists() or partial_root.exists():
        raise MaprayUploadError(
            f"split output already exists: {output_root} or {partial_root}"
        )

    chunks = build_split_chunks(analysis, max_textures=max_textures)
    object_to_chunk = {
        object_index: chunk.index - 1
        for chunk in chunks
        for object_index in chunk.city_objects
    }
    texture_to_chunks: dict[str, list[int]] = {}
    for chunk in chunks:
        for uri in chunk.texture_uris:
            texture_to_chunks.setdefault(uri, []).append(chunk.index - 1)

    partial_root.mkdir(parents=True)
    directories: list[Path] = []
    gml_paths: list[Path] = []
    writers = []
    try:
        root_start = _start_tag(
            analysis.root_tag,
            analysis.root_attributes,
            analysis.namespaces,
            declare_namespaces=True,
        )
        for chunk in chunks:
            directory = partial_root / f"chunk-{chunk.index:03d}"
            directory.mkdir()
            gml_path = directory / f"shibuya-textured-{chunk.index:03d}.gml"
            writer = gml_path.open("w", encoding="utf-8", newline="\n")
            writer.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            writer.write(root_start + "\n")
            writer.write(analysis.bounded_by_xml.rstrip() + "\n")
            directories.append(directory)
            gml_paths.append(gml_path)
            writers.append(writer)

        object_index = 0
        for event, element in ET.iterparse(analysis.source, events=("end",)):
            if element.tag == CITY_OBJECT_MEMBER:
                writer = writers[object_to_chunk[object_index]]
                writer.write(ET.tostring(element, encoding="unicode").rstrip() + "\n")
                object_index += 1
                element.clear()
            elif element.tag == SURFACE_DATA_MEMBER:
                element.clear()

        appearance_member_start = _start_tag(
            f"{{{APP_NS}}}appearanceMember", (), analysis.namespaces
        )
        appearance_start = _start_tag(
            APPEARANCE, analysis.appearance_attributes, analysis.namespaces
        )
        for writer in writers:
            writer.write(appearance_member_start + "\n")
            writer.write(appearance_start + "\n")
            writer.write(analysis.theme_xml.rstrip() + "\n")

        for event, element in ET.iterparse(analysis.source, events=("end",)):
            if element.tag == CITY_OBJECT_MEMBER:
                element.clear()
            elif element.tag == SURFACE_DATA_MEMBER:
                image = element.find(f".//{IMAGE_URI}")
                uri = _normalized_uri(image.text or "") if image is not None else ""
                for chunk_index in texture_to_chunks.get(uri, []):
                    writers[chunk_index].write(
                        _surface_for_chunk(
                            element, chunks[chunk_index], analysis.geometry_owner
                        ).rstrip()
                        + "\n"
                    )
                element.clear()

        for writer in writers:
            writer.write(_end_tag(APPEARANCE, analysis.namespaces) + "\n")
            writer.write(
                _end_tag(f"{{{APP_NS}}}appearanceMember", analysis.namespaces) + "\n"
            )
            writer.write(_end_tag(analysis.root_tag, analysis.namespaces) + "\n")
            writer.close()
        writers.clear()

        source_root = analysis.source.parent
        for chunk, directory in zip(chunks, directories):
            for uri in sorted(chunk.texture_uris):
                relative = Path(uri.replace("/", "\\"))
                source_texture = (source_root / relative).resolve()
                try:
                    source_texture.relative_to(source_root)
                except ValueError as error:
                    raise MaprayUploadError(f"unsafe texture path: {uri}") from error
                if not source_texture.is_file():
                    raise MaprayUploadError(f"referenced texture is missing: {uri}")
                destination = directory / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_texture, destination)

        summaries: list[dict[str, object]] = []
        for chunk, directory in zip(chunks, directories):
            plan = build_upload_plan(directory, batch_size=1000)
            if len(plan.batches) != 1:
                raise MaprayUploadError(
                    f"generated chunk {chunk.index} exceeds one Mapray upload request"
                )
            summaries.append(
                {
                    "index": chunk.index,
                    "directory": directory.name,
                    "city_object_count": len(chunk.city_objects),
                    **plan.as_dict(),
                }
            )

        manifest: dict[str, object] = {
            "schema_version": 1,
            "source": str(analysis.source),
            "max_textures": max_textures,
            "city_object_count": len(analysis.city_ids),
            "unique_texture_count": len(analysis.textures),
            "chunks": summaries,
        }
        (partial_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        partial_root.replace(output_root)
        for summary in summaries:
            summary["root"] = str(output_root / str(summary["directory"]))
        return manifest
    finally:
        for writer in writers:
            writer.close()
