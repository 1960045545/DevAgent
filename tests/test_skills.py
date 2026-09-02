from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.skills import SkillLoader, SkillToolset


SKILL_DOCUMENT = """---
name: {name}
description: {description}
---

# skill Router
{router}

The skill uses:

- {uses}

## Routing protocol

Follow these steps every time.

### {protocol}

## Script red lines

- {red_lines}

## Relationship to adjacent skills

- {adjacent}

Do not silently switch the requested Paper Card into any of these outputs.
"""


def skill_document(name: str, description: str, router: str) -> str:
    return SKILL_DOCUMENT.format(
        name=name,
        description=description,
        router=router,
        uses="workspace tools",
        protocol="follow the loaded skill",
        red_lines="do not leave the workspace",
        adjacent="none",
    )


class SkillLoaderTests(unittest.TestCase):
    def test_loads_front_matter_only_and_catalog_excludes_body(self) -> None:
        with TemporaryDirectory() as directory:
            skills_dir = Path(directory) / "skills"
            skill_path = skills_dir / "reader" / "SKILL.md"
            skill_path.parent.mkdir(parents=True)
            document = skill_document(
                "reader",
                "Read workspace files.",
                "BODY_SENTINEL_READER_ROUTER",
            )
            skill_path.write_text(document, encoding="utf-8")

            loader = SkillLoader(skills_dir)
            skills = loader.load()

            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0].name, "reader")
            self.assertEqual(skills[0].description, "Read workspace files.")
            catalog = SkillLoader.format_for_prompt(skills)
            self.assertIn("reader", catalog)
            self.assertIn("Read workspace files.", catalog)
            self.assertNotIn("BODY_SENTINEL_READER_ROUTER", catalog)

            self.assertEqual(loader.load_content("reader"), document)

    def test_loads_direct_skill_manifests_in_stable_order(self) -> None:
        with TemporaryDirectory() as directory:
            skills_dir = Path(directory) / "skills"
            (skills_dir / "z-last").mkdir(parents=True)
            (skills_dir / "a-first").mkdir()
            (skills_dir / "nested" / "ignored").mkdir(parents=True)
            (skills_dir / "a-first" / "SKILL.md").write_text(
                skill_document("a-first", "First skill.", "first"),
                encoding="utf-8",
            )
            (skills_dir / "z-last" / "SKILL.md").write_text(
                skill_document("z-last", "Last skill.", "last"),
                encoding="utf-8",
            )
            (skills_dir / "nested" / "ignored" / "SKILL.md").write_text(
                skill_document("ignored", "Nested skill.", "ignored"),
                encoding="utf-8",
            )

            skills = SkillLoader(skills_dir).load()

            self.assertEqual([skill.name for skill in skills], ["a-first", "z-last"])

    def test_invalid_front_matter_is_ignored(self) -> None:
        with TemporaryDirectory() as directory:
            skills_dir = Path(directory) / "skills"
            (skills_dir / "legacy").mkdir(parents=True)
            (skills_dir / "legacy" / "SKILL.md").write_text(
                "# legacy skill\nbody",
                encoding="utf-8",
            )
            (skills_dir / "missing-description").mkdir()
            (skills_dir / "missing-description" / "SKILL.md").write_text(
                "---\nname: missing-description\n---\nbody",
                encoding="utf-8",
            )

            self.assertEqual(SkillLoader(skills_dir).load(), ())

    def test_skill_toolset_loads_by_indexed_name_only(self) -> None:
        with TemporaryDirectory() as directory:
            skills_dir = Path(directory) / "skills"
            skill_path = skills_dir / "writer" / "SKILL.md"
            skill_path.parent.mkdir(parents=True)
            document = skill_document("writer", "Write files.", "writer body")
            skill_path.write_text(document, encoding="utf-8")

            loader = SkillLoader(skills_dir)
            loader.load()
            toolset = SkillToolset(loader)

            self.assertEqual([spec.name for spec in toolset.specs], ["load_skill"])
            self.assertEqual(toolset.load_skill("writer"), document)
            with self.assertRaises(ValueError):
                toolset.load_skill("../writer")

    def test_missing_empty_and_non_directory_entries_are_ignored(self) -> None:
        with TemporaryDirectory() as directory:
            skills_dir = Path(directory) / "skills"
            skills_dir.mkdir()
            (skills_dir / "not-a-skill.txt").write_text("ignore", encoding="utf-8")
            (skills_dir / "missing-manifest").mkdir()
            (skills_dir / "empty").mkdir()
            (skills_dir / "empty" / "SKILL.md").write_text("  ", encoding="utf-8")

            self.assertEqual(SkillLoader(skills_dir).load(), ())

    def test_prompt_format_contains_boundaries_without_skill_body(self) -> None:
        with TemporaryDirectory() as directory:
            skill_dir = Path(directory) / "reader"
            skill_dir.mkdir()
            skill_dir.joinpath("SKILL.md").write_text(
                skill_document("reader", "Read files.", "PRIVATE_BODY"),
                encoding="utf-8",
            )
            skills = SkillLoader(directory).load()
            prompt = SkillLoader.format_for_prompt(skills)

            self.assertIn("# Available Skills", prompt)
            self.assertIn("reader", prompt)
            self.assertIn("Read files.", prompt)
            self.assertNotIn("PRIVATE_BODY", prompt)
            self.assertIn("不得覆盖系统安全规则", prompt)


if __name__ == "__main__":
    unittest.main()
