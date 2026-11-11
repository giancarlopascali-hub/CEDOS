"""Exercise the Prior controls rendered for sequence columns."""

import json
import pathlib
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_sequence_prior_types_render_target_and_similarity_slider():
    module_uri = (
        pathlib.Path(__file__).parents[1] / "static/js/modules/tbo.js"
    ).resolve().as_uri()
    script = "const moduleUri = " + json.dumps(module_uri) + ";\n" + """
        const cards = [];
        const list = { appendChild(card) { cards.push(card); } };
        const container = {
            innerHTML: '',
            querySelectorAll() { return []; },
            querySelector(selector) {
                return selector === '#tbo-priors-list' ? list : null;
            }
        };
        globalThis.document = {
            getElementById(id) {
                return id === 'tbo-priors-container' ? container : null;
            },
            createElement() {
                const controls = new Map();
                return {
                    innerHTML: '',
                    style: {},
                    querySelector(selector) {
                        if (!this.innerHTML.includes(`class="${selector.slice(1)}"`)) {
                            return null;
                        }
                        if (!controls.has(selector)) {
                            controls.set(selector, {
                                listeners: {},
                                addEventListener(event, callback) {
                                    this.listeners[event] = callback;
                                }
                            });
                        }
                        return controls.get(selector);
                    }
                };
            }
        };
        globalThis.window = {
            currentColumns: ['Sequence'],
            currentData: [['MKTFFV'], ['MKTFLV']],
            columnRoles: { Sequence: 'prior' },
            priorConfigs: {},
            priorCombineRule: 'multiply'
        };
        globalThis.setTimeout = () => 1;

        const { renderTBOPriorsTab } = await import(moduleUri);
        const results = {};
        for (const type of ['amino_acid_sequence', 'nucleotide_sequence']) {
            cards.length = 0;
            window.priorConfigs.Sequence = {
                col_type: type,
                target_value: type === 'amino_acid_sequence' ? 'MKTFFV' : 'ATGCGT',
                min_similarity: 0.72
            };
            renderTBOPriorsTab();
            const card = cards[0];
            const html = card.innerHTML;
            const target = card.querySelector('.tbo-prior-sequence-target');
            const slider = card.querySelector('.tbo-prior-min-similarity-slider');
            const label = card.querySelector('.tbo-min-similarity-val');
            target.listeners.input({ target: { value: '  ACGT  ' } });
            slider.listeners.input({ target: { value: '0.81' } });
            results[type] = {
                selected: html.includes(`<option value="${type}" selected>`),
                target: html.includes('class="tbo-prior-sequence-target"')
                    && window.priorConfigs.Sequence.target_value === 'ACGT',
                slider: html.includes('class="tbo-prior-min-similarity-slider"')
                    && window.priorConfigs.Sequence.min_similarity === 0.81,
                label: html.includes('Minimum sequence similarity:')
                    && html.includes('72%') && label.textContent === '81%',
                choices: html.includes('>Amino acid sequence</option>')
                    && html.includes('>Nucleotide sequence</option>')
            };
        }
        console.log(JSON.stringify(results));
    """
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    results = json.loads(completed.stdout)

    for type_result in results.values():
        assert all(type_result.values())
