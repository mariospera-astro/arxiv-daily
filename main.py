import logging
import arxiv
import os
from utils import to_timezone_time, get_llm_json_response, send_email, load_processed_ids, append_processed_ids
from paper import Paper
from prompts import recommender_system_prompt, recommender_user_prompt
import json
from pathlib import Path
from construct_pdf import construct_md_file, construct_pdf_file

from settings import load_settings
settings = load_settings(Path("pyproject.toml"))[1]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="./logs/app.log",
    filemode="a",
)


def get_arxiv_papers() -> list[Paper]:
    processed_ids = load_processed_ids()
    papers = []
    client = arxiv.Client()
    search = arxiv.Search(
        query=settings.query,
        max_results=settings.max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    for result in client.results(search):
        publish_date = result.published
        publish_date_bj = to_timezone_time(publish_date, settings.timezone)
        paper = Paper(
            ID=result.entry_id.split('/')[-1],
            title=result.title,
            authors=[author.name for author in result.authors],
            publish_date=publish_date_bj,
            link=result.entry_id,
            abstract=result.summary,
            journal_ref=result.journal_ref
        )
        if paper.ID in processed_ids:
            logging.info(f"Skip already processed paper: {paper.ID}")
            continue
        papers.append(paper)
    return papers


def get_recommend_papers(papers: list[Paper]) -> dict[str, list[tuple[Paper, str]]]:
    recommend_papers = {}
    user_research_interests = ", ".join(settings.research_interests)
    paper_info = ""
    for paper in papers:
        title = paper.title
        abstract = paper.abstract
        paper_id = paper.ID
        paper_info += f"""
Paper ID: {paper_id}
Title: {title}
Abstract: {abstract}
"""

    logging.info("Generating recommendations using LLM.")
    response = get_llm_json_response(
        system_prompt=recommender_system_prompt,
        user_prompt=recommender_user_prompt.format(
            user_interests=user_research_interests,
            paper_info=paper_info
        )
    )
    logging.info(f"Received response from LLM: {response}")
    if response:
        if not response:
        raise RuntimeError("LLM returned empty response (None/empty string).")

    try:
        recommend_obj = json.loads(response)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM did not return valid JSON. Raw response:\n{response}") from e

    # Must be a JSON array
    if not isinstance(recommend_obj, list):
        raise RuntimeError(
            f"LLM response must be a JSON array, got {type(recommend_obj).__name__}. Raw response:\n{response}"
        )

    # Validate each item
    for idx, item in enumerate(recommend_obj):
        if not isinstance(item, dict):
            raise RuntimeError(
                f"LLM array items must be objects, item {idx} is {type(item).__name__}. Raw response:\n{response}"
            )

        paper_id = item.get("paper_id")
        category = item.get("category")
        reason = item.get("reason", "")

        if not isinstance(paper_id, str) or not paper_id.strip():
            raise RuntimeError(f"Missing/invalid paper_id at item {idx}. Raw response:\n{response}")
        if not isinstance(category, str) or not category.strip():
            raise RuntimeError(f"Missing/invalid category at item {idx}. Raw response:\n{response}")
        if not isinstance(reason, str):
            raise RuntimeError(f"Invalid reason at item {idx} (must be string). Raw response:\n{response}")

        category_l = category.lower()

        # match paper by id
        matched = False
        for paper in papers:
            if paper.ID == paper_id:
                recommend_papers.setdefault(category_l, []).append((paper, reason))
                matched = True
                break

        if not matched:
            raise RuntimeError(
                f"LLM returned paper_id '{paper_id}' not present in fetched papers. Raw response:\n{response}"
            )
        except json.JSONDecodeError:
            logging.error("Failed to decode answer from LLM response.")
    return recommend_papers


if __name__ == "__main__":
    logging.info("Starting arXiv Daily Paper Recommendation Process")
    all_papers = get_arxiv_papers()
    logging.info(f"Fetched {len(all_papers)} papers from arXiv.")
    if not all_papers:
        logging.info("All latest papers have been processed before. Skip and exit.")
        exit(0)
    recommended_papers = get_recommend_papers(all_papers)
    logging.info(
        f"Recommended {len(recommended_papers)} papers based on research interests.")
    logging.info("Process completed.")
    if not recommended_papers:
        logging.info("No new recommended papers (all filtered or none matched).")
    md_file_path = construct_md_file(recommended_papers=recommended_papers)
    pdf_file_path = None
    try:
        pdf_file_path = construct_pdf_file(md_file_path=md_file_path)
    except Exception as e:
        logging.warning(f"PDF generation failed: {e}")
    attachment_paths = [md_file_path]
    if pdf_file_path is not None:
        attachment_paths.append(pdf_file_path)
    send_email(attachment_paths=attachment_paths)
    all_ids = [p.ID for p in all_papers]
    if all_ids:
        append_processed_ids(all_ids)
