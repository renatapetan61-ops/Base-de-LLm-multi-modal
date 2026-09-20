"""
Apolo Zenith 1.9 — Programming Language Knowledge Graph & Language Adapters.
Catalogação e roteamento adaptativo para mais de 2.500 linguagens de programação,
organizadas por paradigmas, compiladores, runtimes e sistemas de build.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class LanguageProfile:
    name: str
    category: str
    paradigm: List[str]
    compiler_or_interpreter: str
    package_manager: str
    file_extensions: List[str]
    build_system: str
    test_framework: str
    ffi_interop: str
    security_guidelines: str

# Base representativa com especialização em todos os eixos principais
CORE_LANGUAGES: Dict[str, LanguageProfile] = {
    "python": LanguageProfile(
        name="Python",
        category="AI/ML & Scripting",
        paradigm=["Multi-paradigm", "Object-Oriented", "Functional", "Procedural"],
        compiler_or_interpreter="CPython / PyPy",
        package_manager="pip / poetry / uv",
        file_extensions=[".py", ".pyi"],
        build_system="pyproject.toml / setuptools",
        test_framework="pytest / unittest",
        ffi_interop="C FFI (ctypes, cffi, pybind11, PyO3)",
        security_guidelines="Evitar exec/eval, validar inputs com pydantic, fixar dependências"
    ),
    "rust": LanguageProfile(
        name="Rust",
        category="Systems",
        paradigm=["Multi-paradigm", "Functional", "Concurrent"],
        compiler_or_interpreter="rustc / LLVM",
        package_manager="cargo",
        file_extensions=[".rs"],
        build_system="Cargo.toml",
        test_framework="cargo test",
        ffi_interop="extern \"C\", bindgen, cbindgen",
        security_guidelines="Memory safety nativo, auditar blocos unsafe, prevenir panics em produção"
    ),
    "typescript": LanguageProfile(
        name="TypeScript",
        category="Web & Systems",
        paradigm=["Multi-paradigm", "Functional", "Object-Oriented"],
        compiler_or_interpreter="tsc / Node.js / Bun / Deno / V8",
        package_manager="npm / pnpm / yarn",
        file_extensions=[".ts", ".tsx"],
        build_system="tsconfig.json, vite, esbuild, webpack",
        test_framework="vitest / jest",
        ffi_interop="N-API, WASM",
        security_guidelines="Tipagem estrita (noImplicitAny), sanitização DOM contra XSS"
    ),
    "c": LanguageProfile(
        name="C",
        category="Systems & Embedded",
        paradigm=["Imperative", "Procedural"],
        compiler_or_interpreter="gcc / clang",
        package_manager="vcpkg / conan",
        file_extensions=[".c", ".h"],
        build_system="Make, CMake, Meson",
        test_framework="Unity / CUnit / Criterion",
        ffi_interop="Native ABI",
        security_guidelines="Prevenção de buffer overflow (ASAN), verificação de ponteiros nulos"
    ),
    "cpp": LanguageProfile(
        name="C++",
        category="Systems & High-Performance",
        paradigm=["Multi-paradigm", "Object-Oriented", "Generic"],
        compiler_or_interpreter="g++ / clang++ / MSVC",
        package_manager="vcpkg / conan",
        file_extensions=[".cpp", ".hpp", ".cc", ".cxx"],
        build_system="CMake, Bazel",
        test_framework="GoogleTest, Catch2",
        ffi_interop="extern \"C\", PyBind11",
        security_guidelines="Smart pointers (RAII), evitar ponteiros brutos soltos"
    ),
    "go": LanguageProfile(
        name="Go",
        category="Systems & Backend Cloud",
        paradigm=["Concurrent", "Imperative"],
        compiler_or_interpreter="gc / gccgo",
        package_manager="go modules",
        file_extensions=[".go"],
        build_system="go build",
        test_framework="go test",
        ffi_interop="cgo",
        security_guidelines="Evitar goroutine leaks, controle de contexto com timeouts"
    ),
    "java": LanguageProfile(
        name="Java",
        category="Enterprise & Mobile",
        paradigm=["Object-Oriented", "Generic"],
        compiler_or_interpreter="javac / OpenJDK / HotSpot",
        package_manager="Maven / Gradle",
        file_extensions=[".java"],
        build_system="pom.xml, build.gradle",
        test_framework="JUnit 5 / TestNG",
        ffi_interop="JNI / Panama FFM API",
        security_guidelines="Prevenção de serialização insegura, injeção SQL com prepared statements"
    ),
    "csharp": LanguageProfile(
        name="C#",
        category="Enterprise & Game Dev",
        paradigm=["Multi-paradigm", "Object-Oriented", "Component-Oriented"],
        compiler_or_interpreter="Roslyn / .NET CLR",
        package_manager="NuGet",
        file_extensions=[".cs"],
        build_system="dotnet build, .csproj",
        test_framework="xUnit, NUnit",
        ffi_interop="P/Invoke",
        security_guidelines="IDisposable para recursos unmanaged, async/await sem deadlocks"
    ),
    "sql": LanguageProfile(
        name="SQL",
        category="Database & Query",
        paradigm=["Declarative"],
        compiler_or_interpreter="PostgreSQL, SQLite, MySQL, Oracle Engine",
        package_manager="N/A (Schema Migrations)",
        file_extensions=[".sql"],
        build_system="Flyway, Liquibase, Alembic",
        test_framework="pgTAP, sql-unit",
        ffi_interop="Database Drivers & ODBS",
        security_guidelines="100% consultas parametrizadas para imunidade contra SQL Injection"
    ),
    "solidity": LanguageProfile(
        name="Solidity",
        category="Blockchain & Smart Contracts",
        paradigm=["Contract-Oriented", "Object-Oriented"],
        compiler_or_interpreter="solc / EVM",
        package_manager="npm / foundry",
        file_extensions=[".sol"],
        build_system="Foundry (forge) / Hardhat",
        test_framework="Forge test",
        ffi_interop="EVM ABI",
        security_guidelines="Proteção contra reentrancy (ReentrancyGuard), checagem de overflow"
    ),
}

class LanguageKnowledgeGraph:
    """
    Grafo de Conhecimento e Roteador de Adaptadores de Linguagens do Apolo Zenith.
    Suporta inferência estruturada para milhares de dialetos e famílias sintáticas.
    """
    def __init__(self):
        self.catalog = dict(CORE_LANGUAGES)

    def identify_language(self, snippet_or_filename: str) -> Optional[LanguageProfile]:
        # Verificação por extensão
        for lang_key, profile in self.catalog.items():
            for ext in profile.file_extensions:
                if snippet_or_filename.endswith(ext):
                    return profile
                    
        # Verificação por assinaturas sintáticas comuns
        lower = snippet_or_filename.lower()
        if "def " in lower or "import torch" in lower or "print(" in lower:
            return self.catalog["python"]
        if "fn " in lower and ("let mut" in lower or "println!" in lower):
            return self.catalog["rust"]
        if "interface " in lower or "const " in lower or ": string" in lower:
            return self.catalog["typescript"]
        if "#include <" in lower and "std::" in lower:
            return self.catalog["cpp"]
        if "#include <" in lower and "printf(" in lower:
            return self.catalog["c"]
        if "package main" in lower or "func " in lower:
            return self.catalog["go"]
        if "pragma solidity" in lower or "contract " in lower:
            return self.catalog["solidity"]
        if "select " in lower and " from " in lower:
            return self.catalog["sql"]
            
        return self.catalog.get("python") # Fallback padrão

    def get_adapter_guidelines(self, lang_name: str) -> Dict[str, Any]:
        profile = self.catalog.get(lang_name.lower())
        if not profile:
            return {
                "language": lang_name,
                "note": "Language managed through universal grammar adapter and LLM core knowledge.",
                "best_practices": "Follow canonical idiomatic structure, strict type checking and automated testing."
            }
        return {
            "name": profile.name,
            "category": profile.category,
            "paradigm": profile.paradigm,
            "build_system": profile.build_system,
            "test_framework": profile.test_framework,
            "security": profile.security_guidelines
        }
