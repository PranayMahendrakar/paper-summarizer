#!/usr/bin/env python3
"""
Scientific Paper Summarizer - Llama-Based Research Tool
Creates concise, accurate summaries of complex scientific papers
Author: Pranay M
"""

import ollama
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.markdown import Markdown
import json

console = Console()

PAPER_SECTIONS = ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion"]
SUMMARY_TYPES = ["lay_summary", "technical_summary", "executive_brief", "tweet_thread", "blog_post"]

class ScientificPaperSummarizer:
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.field = "General Science"
        self.audience_level = "graduate"
    
    def summarize_paper(self, content: str, summary_type: str = "technical_summary") -> dict:
        length_guide = {
            "lay_summary": "300 words, no jargon",
            "technical_summary": "500 words, field-appropriate terminology",
            "executive_brief": "200 words, key findings only",
            "tweet_thread": "10 tweets, 280 characters each",
            "blog_post": "800 words, engaging style"
        }
        
        prompt = f"""Summarize this scientific paper as a {summary_type}.

Paper:
{content}

Requirements: {length_guide.get(summary_type, 'standard length')}
Field: {self.field}
Audience: {self.audience_level} level

Return JSON:
{{
    "title": "paper title",
    "authors_inferred": "if visible",
    "field": "research field",
    "summary_type": "{summary_type}",
    "summary": "the summary text",
    "key_findings": ["finding 1", "finding 2", "finding 3"],
    "methodology_brief": "methods used",
    "significance": "why this matters",
    "limitations": ["limitation 1"],
    "future_directions": ["next steps suggested"],
    "keywords": ["key terms"],
    "reading_time": "X minutes"
}}"""

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return self._parse_json(response['message']['content'])
    
    def extract_key_findings(self, content: str) -> dict:
        prompt = f"""Extract and structure the key findings from this scientific paper.

Paper:
{content}

Return JSON:
{{
    "primary_findings": [
        {{
            "finding": "description",
            "evidence": "supporting data",
            "statistical_significance": "p-value or CI if available",
            "effect_size": "magnitude of effect"
        }}
    ],
    "secondary_findings": ["additional findings"],
    "negative_results": ["null findings if any"],
    "unexpected_findings": ["surprising results"],
    "findings_hierarchy": "which finding is most important and why"
}}"""

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return self._parse_json(response['message']['content'])
    
    def analyze_methodology(self, content: str) -> dict:
        prompt = f"""Analyze the methodology of this scientific paper.

Paper:
{content}

Return JSON:
{{
    "study_design": "type of study",
    "sample": {{
        "size": "N",
        "characteristics": "population details",
        "recruitment": "how selected"
    }},
    "data_collection": ["methods used"],
    "analysis_methods": ["statistical/analytical techniques"],
    "controls": ["control measures"],
    "variables": {{
        "independent": ["variables"],
        "dependent": ["outcomes"],
        "confounders": ["controlled for"]
    }},
    "strengths": ["methodological strengths"],
    "weaknesses": ["potential issues"],
    "reproducibility": "assessment of reproducibility"
}}"""

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return self._parse_json(response['message']['content'])
    
    def generate_visual_abstract(self, content: str) -> dict:
        prompt = f"""Create a visual abstract description for this paper.

Paper:
{content}

Return JSON:
{{
    "title_block": "paper title",
    "graphical_elements": [
        {{
            "element": "description of visual",
            "position": "where in layout",
            "purpose": "what it shows"
        }}
    ],
    "flow": "left-to-right/top-to-bottom narrative",
    "key_takeaway_box": "main message",
    "icons_suggested": ["relevant icons"],
    "color_scheme": "suggested colors",
    "text_elements": ["text to include"]
}}"""

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return self._parse_json(response['message']['content'])
    
    def compare_papers(self, papers: list) -> dict:
        papers_text = "\n\n---PAPER BREAK---\n\n".join([f"PAPER {i+1}:\n{p}" for i, p in enumerate(papers)])
        
        prompt = f"""Compare these scientific papers.

{papers_text}

Return JSON:
{{
    "papers_overview": [
        {{
            "paper_number": 1,
            "title_inferred": "title",
            "main_finding": "key finding"
        }}
    ],
    "methodological_comparison": {{
        "similarities": ["shared methods"],
        "differences": ["different approaches"]
    }},
    "findings_comparison": {{
        "agreements": ["consistent findings"],
        "contradictions": ["conflicting findings"],
        "complementary": ["findings that build on each other"]
    }},
    "quality_comparison": {{
        "sample_sizes": ["N for each"],
        "rigor_assessment": ["assessment for each"]
    }},
    "synthesis": "what we learn from combining these papers",
    "research_gaps": ["gaps still remaining"]
}}"""

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return self._parse_json(response['message']['content'])
    
    def generate_citation_summary(self, content: str) -> str:
        prompt = f"""Create a citation-ready summary paragraph for this paper.

Paper:
{content}

Write a 2-3 sentence summary suitable for citing in a literature review.
Include: authors (if visible), main finding, methodology type, and significance.
Use academic writing style."""

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response['message']['content']
    
    def identify_implications(self, content: str) -> dict:
        prompt = f"""Identify the implications of this research.

Paper:
{content}

Return JSON:
{{
    "theoretical_implications": [
        {{
            "implication": "description",
            "affected_theories": ["theories impacted"]
        }}
    ],
    "practical_implications": [
        {{
            "implication": "description",
            "stakeholders": ["who benefits"],
            "implementation": "how to apply"
        }}
    ],
    "policy_implications": ["policy relevance"],
    "future_research": [
        {{
            "direction": "research direction",
            "rationale": "why needed",
            "approach": "suggested methodology"
        }}
    ],
    "limitations_affecting_implications": ["caveats"]
}}"""

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return self._parse_json(response['message']['content'])
    
    def simplify_for_public(self, content: str) -> str:
        prompt = f"""Rewrite this scientific paper content for a general public audience.

Paper:
{content}

Requirements:
1. No jargon - explain all technical terms
2. Use analogies and examples
3. Focus on "why should I care?"
4. Keep it engaging and accessible
5. About 400 words

Write as if explaining to a curious friend with no science background."""

        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response['message']['content']
    
    def _parse_json(self, content: str) -> dict:
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except:
            pass
        return {"raw_response": content}


def display_menu():
    table = Table(title="📄 Scientific Paper Summarizer", show_header=True)
    table.add_column("Option", style="cyan", width=6)
    table.add_column("Feature", style="green")
    table.add_column("Description", style="white")
    
    table.add_row("1", "Summarize Paper", "Create paper summary")
    table.add_row("2", "Key Findings", "Extract main findings")
    table.add_row("3", "Methodology", "Analyze methods")
    table.add_row("4", "Visual Abstract", "Create visual abstract plan")
    table.add_row("5", "Compare Papers", "Compare multiple papers")
    table.add_row("6", "Citation Summary", "Generate citable summary")
    table.add_row("7", "Implications", "Identify research implications")
    table.add_row("8", "Public Version", "Simplify for general audience")
    table.add_row("9", "Settings", "Set field and audience level")
    table.add_row("0", "Exit", "Close application")
    
    console.print(table)


def main():
    console.print(Panel.fit(
        "[bold blue]📄 Scientific Paper Summarizer[/bold blue]\n"
        "[green]AI-Powered Research Paper Analysis[/green]\n"
        "[dim]Author: Pranay M[/dim]",
        border_style="blue"
    ))
    
    summarizer = ScientificPaperSummarizer()
    
    while True:
        display_menu()
        console.print(f"[dim]Field: {summarizer.field} | Audience: {summarizer.audience_level}[/dim]")
        
        choice = Prompt.ask("\n[cyan]Select option[/cyan]", default="0")
        
        if choice == "0":
            console.print("[yellow]Goodbye! Keep researching! 📄[/yellow]")
            break
        
        elif choice == "9":
            summarizer.field = Prompt.ask("Research field", default="General Science")
            summarizer.audience_level = Prompt.ask("Audience", 
                choices=["undergraduate", "graduate", "expert", "general"], default="graduate")
            console.print("[green]✓ Settings updated[/green]")
            continue
        
        # Get paper content
        if choice in ["1", "2", "3", "4", "6", "7", "8"]:
            console.print("[dim]Paste paper content (end with 'EOF'):[/dim]")
            lines = []
            while True:
                line = input()
                if line.strip() == "EOF":
                    break
                lines.append(line)
            content = "\n".join(lines)
        
        with console.status("[bold green]Processing..."):
            if choice == "1":
                console.print("[bold]Summary types:[/bold]")
                for st in SUMMARY_TYPES:
                    console.print(f"  - {st}")
                summary_type = Prompt.ask("Type", default="technical_summary")
                result = summarizer.summarize_paper(content, summary_type)
                console.print(Panel(Markdown(f"```json\n{json.dumps(result, indent=2)}\n```"),
                                   title="📝 Paper Summary"))
            
            elif choice == "2":
                findings = summarizer.extract_key_findings(content)
                console.print(Panel(Markdown(f"```json\n{json.dumps(findings, indent=2)}\n```"),
                                   title="🔬 Key Findings"))
            
            elif choice == "3":
                methods = summarizer.analyze_methodology(content)
                console.print(Panel(Markdown(f"```json\n{json.dumps(methods, indent=2)}\n```"),
                                   title="🔧 Methodology Analysis"))
            
            elif choice == "4":
                visual = summarizer.generate_visual_abstract(content)
                console.print(Panel(Markdown(f"```json\n{json.dumps(visual, indent=2)}\n```"),
                                   title="🎨 Visual Abstract"))
            
            elif choice == "5":
                papers = []
                num = IntPrompt.ask("How many papers to compare", default=2)
                for i in range(num):
                    console.print(f"[dim]Paste Paper {i+1} (end with 'EOF'):[/dim]")
                    lines = []
                    while True:
                        line = input()
                        if line.strip() == "EOF":
                            break
                        lines.append(line)
                    papers.append("\n".join(lines))
                comparison = summarizer.compare_papers(papers)
                console.print(Panel(Markdown(f"```json\n{json.dumps(comparison, indent=2)}\n```"),
                                   title="⚖️ Paper Comparison"))
            
            elif choice == "6":
                citation = summarizer.generate_citation_summary(content)
                console.print(Panel(citation, title="📚 Citation Summary"))
            
            elif choice == "7":
                implications = summarizer.identify_implications(content)
                console.print(Panel(Markdown(f"```json\n{json.dumps(implications, indent=2)}\n```"),
                                   title="💡 Implications"))
            
            elif choice == "8":
                public = summarizer.simplify_for_public(content)
                console.print(Panel(Markdown(public), title="👥 Public Summary"))
        
        console.print("\n" + "="*50)


if __name__ == "__main__":
    main()
