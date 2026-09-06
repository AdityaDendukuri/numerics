/// @file test_concept_hierarchy.cpp
/// @brief The concept hierarchy is an architectural invariant, checked like any other.
///
/// `numerics` claims that mathematical structure is enforced at compile time. That claim
/// is only worth something if the concepts actually form a hierarchy — if each one is a
/// refinement of something rather than a predicate asserted against raw syntax. Left to
/// convention, a tower decays into a flat pile: every new domain adds a bare `requires`
/// block, and after enough of them the vocabulary has a hundred entries and no shape.
///
/// So the shape is asserted here, the way `Package.CoreDependencyFree` asserts that the
/// kernel tier has no dependencies. Two things are checked over every concept the library
/// defines:
///
///   1. **Rootedness.** A concept naming a mathematical property must refine another
///      concept. The exceptions are explicit and listed below, each with a reason.
///   2. **Representation independence.** A concept must not name a concrete container
///      (`vec`, `mat`, `spmat`) inside its body. Naming one locks the
///      concept to `double` and to one storage layout, which defeats the point of having
///      an algebraic tower above the kernel at all.
///
/// It also checks that no core concept is defined twice — the defect that let
/// `num::vector_space` and `num::math::vector_space` drift into different predicates
/// wearing the same name.
///
/// This parses the headers rather than compiling them. A `static_assert` can check that
/// one type models one concept; it cannot see the shape of the vocabulary as a whole.

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

/// Concepts allowed to stand at the root of the graph, with the reason each is primitive.
const std::map<std::string, std::string> &allowed_roots() {
    static const std::map<std::string, std::string> roots = {
        // Machinery. These read or describe the claim system; they are not claims.
        {"claims", "the mechanism that reads a type's laws"},
        {"law_tag", "membership in the law lattice"},
        {"mathematical_proposition", "a law that can be attached to a value as evidence"},
        {"tag_invocable", "customization-point detection"},
        {"has_free_axpy", "ADL probe"},
        {"has_free_dot", "ADL probe"},
        {"has_free_norm", "ADL probe"},
        {"has_free_scale", "ADL probe"},
        {"has_square_tag", "tag detection"},
        {"raw_reducible", "contiguity probe"},
        // Storage. Deliberately outside the tower: a statement about memory, not a map.
        {"csr", "storage layout"},
        {"tridiagonal", "storage layout"},
        {"contiguous", "storage layout"},
        {"dense_row_major", "storage layout"},
        {"banded", "storage layout"},
        // Genuine primitives of a different branch of mathematics than linear algebra.
        // Forcing these onto the vector-space tower would be false layering: a union-find
        // is not a vector space, and saying it refined one would make the hierarchy a lie.
        {"field", "the scalar field: root of the algebraic tower"},
        {"square_extent_2d", "root of the discrete index-space family"},
        {"equivalence_relation", "set theory: a partition, no algebraic structure"},
        {"incidence_structure", "graph theory: vertices and edges, no algebraic structure"},
        {"addressable_priority_queue", "order theory, rooted in std::totally_ordered"},
        {"random_engine", "rooted in std::uniform_random_bit_generator"},
        // Closed sets of algorithm tags, matched with std::same_as. Dispatch, not
        // mathematics; there is nothing for them to refine.
        {"is_explicit_ode_alg", "algorithm tag set"},
        {"is_mcmc_alg", "algorithm tag set"},
    };
    return roots;
}

/// Concrete types a concept body must not name, and what naming one costs.
const std::vector<std::string> &concrete_types() {
    static const std::vector<std::string> types = {"vec", "mat", "spmat", "cvec"};
    return types;
}

struct ConceptDef {
    std::string name;
    std::string file;
    std::string body;
};

std::vector<ConceptDef> parse_concepts(const fs::path &include_dir) {
    std::vector<ConceptDef> defs;
    const std::regex decl(R"(concept\s+([A-Za-z_]\w*)\s*=)");
    for (const auto &entry : fs::recursive_directory_iterator(include_dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".hpp") {
            continue;
        }
        std::ifstream in(entry.path());
        std::stringstream buffer;
        buffer << in.rdbuf();
        const std::string source = buffer.str();
        const std::string rel = fs::relative(entry.path(), include_dir).string();

        for (auto it = std::sregex_iterator(source.begin(), source.end(), decl);
             it != std::sregex_iterator(); ++it) {
            // The body runs to the semicolon that closes the definition. Brace depth
            // keeps a `requires { ... ; ... }` block from ending it early.
            std::size_t pos = static_cast<std::size_t>(it->position()) + it->length();
            int depth = 0;
            std::size_t end = pos;
            for (; end < source.size(); ++end) {
                const char c = source[end];
                if (c == '{') { ++depth; }
                else if (c == '}') { --depth; }
                else if (c == ';' && depth == 0) { break; }
            }
            defs.push_back({it->str(1), rel, source.substr(pos, end - pos)});
        }
    }
    return defs;
}

/// True when `body` names `other` in template-argument position, i.e. refines it.
bool refines(const std::string &body, const std::string &other) {
    const std::regex use(R"(\b)" + other + R"(\s*<)");
    return std::regex_search(body, use);
}

fs::path include_root() {
    // The test binary runs from the build tree; the source path is injected by CMake.
    return fs::path(NUMERICS_SOURCE_DIR) / "include";
}

} // namespace

TEST(ConceptHierarchy, EveryMathematicalConceptRefinesAnother) {
    const auto defs = parse_concepts(include_root());
    ASSERT_GT(defs.size(), 50U) << "parser found too few concepts; it is probably broken";

    std::set<std::string> names;
    for (const auto &d : defs) {
        names.insert(d.name);
    }

    std::vector<std::string> unrooted;
    for (const auto &d : defs) {
        if (allowed_roots().count(d.name) != 0) {
            continue;
        }
        const bool rooted = std::any_of(names.begin(), names.end(), [&](const std::string &other) {
            return other != d.name && refines(d.body, other);
        });
        if (!rooted) {
            unrooted.push_back(d.name + "  (" + d.file + ")");
        }
    }

    EXPECT_TRUE(unrooted.empty())
        << "These concepts are asserted against raw syntax instead of refining the tower.\n"
        << "Either state what they refine, or add them to allowed_roots() with the reason\n"
        << "they are genuinely primitive:\n  "
        << [&] {
               std::string s;
               for (const auto &u : unrooted) { s += u + "\n  "; }
               return s;
           }();
}

TEST(ConceptHierarchy, NoConceptIsLockedToAConcreteContainer) {
    const auto defs = parse_concepts(include_root());
    std::vector<std::string> locked;
    for (const auto &d : defs) {
        for (const auto &concrete : concrete_types()) {
            // A defaulted template parameter (`class V = vec`) is fine: it names the
            // common case while leaving the concept open. Naming the type inside the
            // requires-block is what locks it.
            std::string body = d.body;
            const std::regex default_arg(R"(=\s*)" + concrete + R"(\b)");
            body = std::regex_replace(body, default_arg, "= _");
            if (std::regex_search(body, std::regex(R"(\b)" + concrete + R"(\b)"))) {
                locked.push_back(d.name + " names " + concrete + "  (" + d.file + ")");
            }
        }
    }
    EXPECT_TRUE(locked.empty())
        << "A concept that names a concrete container cannot be used with cvec,\n"
        << "std::vector<float>, or any foreign type -- which is the reason the algebraic\n"
        << "tower exists. Take the space as a template parameter refining vector_space,\n"
        << "defaulted to the concrete type so existing call sites keep compiling:\n  "
        << [&] {
               std::string s;
               for (const auto &l : locked) { s += l + "\n  "; }
               return s;
           }();
}

TEST(ConceptHierarchy, CoreConceptsAreDefinedExactlyOnce) {
    const auto defs = parse_concepts(include_root());
    // The tower's own vocabulary. Two definitions of one of these is how
    // num::vector_space and num::math::vector_space became different predicates.
    const std::set<std::string> core = {"vector_space",   "normed_space",    "inner_product_space",
                                        "hilbert_space",  "additive_group",  "linear_map",
                                        "linear_operator", "spd_operator",   "psd_operator",
                                        "self_adjoint_operator", "normal_operator"};
    std::map<std::string, std::vector<std::string>> where;
    for (const auto &d : defs) {
        if (core.count(d.name) != 0) {
            where[d.name].push_back(d.file);
        }
    }
    for (const auto &[name, files] : where) {
        EXPECT_EQ(files.size(), 1U) << name << " is defined in " << files.size()
                                    << " places; the tower must have exactly one of each";
    }
}
