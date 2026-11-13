from . import blog  # noqa: F401
from .glossary import aka_registry, registry


def test_term_list():
    for clsname, obj in sorted(registry.items()):
        for name in obj.pretty_names():
            assert isinstance(name, str)


def test_term_body():
    n_checks = 0
    for clsname, obj in sorted(registry.items()):
        assert isinstance(obj.definition_html, str)
        if obj.code_refs:
            for ref_text, ref_url in obj.code_links.items():
                assert isinstance(ref_text, str)
                assert isinstance(ref_url, str)
                assert ref_url.startswith('http')

        if obj.see_also or obj.as_discussed_in_posts:
            for term, ref_text in sorted(obj.see_also.items()):
                assert term in aka_registry, (clsname, obj, term, ref_text)
                assert isinstance(ref_text, str)
            for post_obj, hashref, descr in obj.as_discussed_in_posts:
                assert isinstance(post_obj.siteref, str)
                assert isinstance(hashref, str)
                assert isinstance(descr, str)
        n_checks += 1
    assert n_checks > 10, n_checks # sanity check, not e.g. 0
