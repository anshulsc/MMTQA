import streamlit as st
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Golden Answer Verification",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .correct-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .incorrect-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .revision-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .stDataFrame {
        border: 2px solid #4CAF50;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


class GoldenAnswerVerifier:
    """Main application class for golden answer verification."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.qa_dir = self.base_dir / "qa_pairs" / "en"
        self.qa_corrected_dir = self.base_dir / "qa_pairs_corrected"
        self.tables_dir = self.base_dir / "tables"
        self.verification_file = self.base_dir / "golden_answer_verification_results_v2.jsonl"
        
        # Logging and checkpoint files
        self.logs_dir = self.base_dir / "verification_logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        self.changes_log_file = self.logs_dir / "changes_log.jsonl"
        self.decisions_log_file = self.logs_dir / "decisions_log.jsonl"
        self.checkpoint_file = self.logs_dir / "checkpoint.json"
        self.summary_log_file = self.logs_dir / "verification_summary.json"
        
        # Create corrected directory if it doesn't exist
        self.qa_corrected_dir.mkdir(exist_ok=True)
        
        # Initialize session state
        if 'current_index' not in st.session_state:
            st.session_state.current_index = 0
        if 'filter_status' not in st.session_state:
            st.session_state.filter_status = "All"
        if 'filter_dataset' not in st.session_state:
            st.session_state.filter_dataset = "All"
        if 'verification_data' not in st.session_state:
            st.session_state.verification_data = None
        if 'changes_made' not in st.session_state:
            st.session_state.changes_made = {}
        if 'decisions_made' not in st.session_state:
            st.session_state.decisions_made = {}
        if 'session_start_time' not in st.session_state:
            st.session_state.session_start_time = datetime.now().isoformat()
        
        # Load existing checkpoint if available
        self.load_checkpoint()
    
    def load_checkpoint(self):
        """Load checkpoint to resume from last position."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                    st.session_state.current_index = checkpoint.get('current_index', 0)
                    st.session_state.filter_status = checkpoint.get('filter_status', 'All')
                    st.session_state.filter_dataset = checkpoint.get('filter_dataset', 'All')
                    st.session_state.changes_made = checkpoint.get('changes_made', {})
                    st.session_state.decisions_made = checkpoint.get('decisions_made', {})
            except Exception as e:
                st.warning(f"Could not load checkpoint: {e}")
    
    def save_checkpoint(self):
        """Save current progress as checkpoint."""
        checkpoint = {
            'current_index': st.session_state.current_index,
            'filter_status': st.session_state.filter_status,
            'filter_dataset': st.session_state.filter_dataset,
            'changes_made': st.session_state.changes_made,
            'decisions_made': st.session_state.decisions_made,
            'timestamp': datetime.now().isoformat(),
            'session_start': st.session_state.session_start_time
        }
        
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    
    def log_decision(self, question_id: str, decision_type: str, details: Dict):
        """Log a verification decision to JSONL file."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'question_id': question_id,
            'decision_type': decision_type,  # 'changed', 'accepted', 'skipped'
            'details': details,
            'session_start': st.session_state.session_start_time
        }
        
        # Append to JSONL log
        with open(self.decisions_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        # Store in session state
        st.session_state.decisions_made[question_id] = log_entry
        
        # Save checkpoint after each decision
        self.save_checkpoint()
    
    def log_change(self, question_id: str, old_answer: Any, new_answer: Any, metadata: Dict):
        """Log an answer change to JSONL file."""
        change_entry = {
            'timestamp': datetime.now().isoformat(),
            'question_id': question_id,
            'old_answer': old_answer,
            'new_answer': new_answer,
            'metadata': metadata,
            'session_start': st.session_state.session_start_time
        }
        
        # Append to changes JSONL log
        with open(self.changes_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(change_entry, ensure_ascii=False) + '\n')
        
        # Store in session state
        st.session_state.changes_made[question_id] = change_entry
    
    def generate_summary_report(self):
        """Generate a summary report of all verification activities."""
        summary = {
            'report_generated': datetime.now().isoformat(),
            'session_start': st.session_state.session_start_time,
            'total_decisions': len(st.session_state.decisions_made),
            'total_changes': len(st.session_state.changes_made),
            'decisions_breakdown': {
                'changed': sum(1 for d in st.session_state.decisions_made.values() if d['decision_type'] == 'changed'),
                'accepted': sum(1 for d in st.session_state.decisions_made.values() if d['decision_type'] == 'accepted'),
                'skipped': sum(1 for d in st.session_state.decisions_made.values() if d['decision_type'] == 'skipped')
            },
            'changes_by_dataset': {},
            'changes_by_status': {},
            'questions_processed': list(st.session_state.decisions_made.keys())
        }
        
        # Analyze changes by dataset and original status
        for qid, change in st.session_state.changes_made.items():
            dataset = change['metadata'].get('dataset', 'unknown')
            status = change['metadata'].get('original_status', 'unknown')
            
            summary['changes_by_dataset'][dataset] = summary['changes_by_dataset'].get(dataset, 0) + 1
            summary['changes_by_status'][status] = summary['changes_by_status'].get(status, 0) + 1
        
        # Save summary
        with open(self.summary_log_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary
    
    def load_verification_results(self) -> List[Dict]:
        """Load verification results from JSONL file."""
        results = []
        if self.verification_file.exists():
            with open(self.verification_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        results.append(json.loads(line.strip()))
                    except Exception as e:
                        st.error(f"Error loading line: {e}")
        return results
    
    def load_table_data(self, question_id: str) -> Dict:
        """Load table data based on question ID."""
        table_id = "_".join(question_id.split("_")[:-1])
        table_file = self.tables_dir / f"{table_id}.json"
        
        if table_file.exists():
            with open(table_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def table_to_dataframe(self, table_data: Dict) -> pd.DataFrame:
        """Convert JSON table to pandas DataFrame."""
        if not table_data or "data" not in table_data:
            return pd.DataFrame()
        
        data = table_data["data"]
        if not data or len(data) == 0:
            return pd.DataFrame()
        
        headers = data[0]
        rows = data[1:]
        
        return pd.DataFrame(rows, columns=headers)
    
    def update_qa_file(self, question_id: str, new_answer: List[str]):
        """Update the answer in the corrected QA file (creates copy if doesn't exist)."""
        table_id = "_".join(question_id.split("_")[:-1])
        original_qa_file = self.qa_dir / f"{table_id}_qa.json"
        corrected_qa_file = self.qa_corrected_dir / f"{table_id}_qa.json"
        
        if not original_qa_file.exists():
            st.error(f"Original QA file not found: {original_qa_file}")
            return False
        
        try:
            if not corrected_qa_file.exists():
                with open(original_qa_file, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
                st.info(f"📋 Creating new corrected file: {corrected_qa_file.name}")
            else:
                with open(corrected_qa_file, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
            
            updated = False
            for item in qa_data:
                if item['question_id'] == question_id:
                    item['answer'] = new_answer
                    updated = True
                    break
            
            if updated:
                with open(corrected_qa_file, 'w', encoding='utf-8') as f:
                    json.dump(qa_data, f, indent=2, ensure_ascii=False)
                return True
            else:
                st.error(f"Question ID not found in QA file: {question_id}")
                return False
                
        except Exception as e:
            st.error(f"Error updating QA file: {e}")
            return False
    
    def get_dataset_type(self, question_id: str) -> str:
        """Extract dataset type from question_id."""
        parts = question_id.lower().split('_')
        if len(parts) > 0:
            dataset = parts[0]
            if 'arxiv' in dataset:
                return 'arxiv'
            elif 'finqa' in dataset:
                return 'finqa'
            elif 'wiki' in dataset or 'wikitq' in dataset:
                return 'wikitq'
            elif 'tat' in dataset:
                return 'tat-qa'
            elif 'multihiertt' in dataset:
                return 'multihiertt'
        return 'unknown'
    
    def get_unique_datasets(self, data: List[Dict]) -> List[str]:
        """Get list of unique dataset types from verification data."""
        datasets = set()
        for item in data:
            dataset = self.get_dataset_type(item['question_id'])
            datasets.add(dataset)
        return sorted(list(datasets))

    def filter_data(self, data: List[Dict], status: str, dataset: str) -> List[Dict]:
        """Filter verification data by status and dataset."""
        filtered = data
        if status != "All":
            filtered = [d for d in filtered if d.get('is_golden_correct') == status]
        if dataset != "All":
            filtered = [d for d in filtered if self.get_dataset_type(d['question_id']) == dataset]
        return filtered
    
    def display_status_badge(self, status: str):
        """Display status badge with appropriate styling."""
        if status == "Correct":
            st.markdown('<div class="correct-box">✅ <b>CORRECT</b></div>', unsafe_allow_html=True)
        elif status == "Incorrect":
            st.markdown('<div class="incorrect-box">❌ <b>INCORRECT</b></div>', unsafe_allow_html=True)
        elif status == "Needs_Revision":
            st.markdown('<div class="revision-box">⚠️ <b>NEEDS REVISION</b></div>', unsafe_allow_html=True)
        else:
            st.error(f"⚡ ERROR: {status}")
    
    def run(self):
        """Run the Streamlit application."""
        st.title("🔍 Golden Answer Verification System")
        st.markdown("---")
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Load data button
            if st.button("🔄 Load/Refresh Data", use_container_width=True):
                with st.spinner("Loading verification results..."):
                    st.session_state.verification_data = self.load_verification_results()
                st.success(f"Loaded {len(st.session_state.verification_data)} results")
            
            st.markdown("---")
            
            # Filter options
            st.subheader("🔍 Filters")
            
            filter_status = st.selectbox(
                "Status",
                ["All", "Incorrect", "Needs_Revision", "Correct", "Error"],
                key="filter_select"
            )
            
            if st.session_state.verification_data:
                datasets = self.get_unique_datasets(st.session_state.verification_data)
                filter_dataset = st.selectbox(
                    "Dataset Type",
                    ["All"] + datasets,
                    key="dataset_filter"
                )
            else:
                filter_dataset = "All"
            
            if (filter_status != st.session_state.filter_status or 
                filter_dataset != st.session_state.filter_dataset):
                st.session_state.filter_status = filter_status
                st.session_state.filter_dataset = filter_dataset
                st.session_state.current_index = 0
            
            st.markdown("---")
            
            # Statistics
            if st.session_state.verification_data:
                st.subheader("📊 Statistics")
                data = st.session_state.verification_data
                
                total = len(data)
                correct = sum(1 for d in data if d['is_golden_correct'] == 'Correct')
                incorrect = sum(1 for d in data if d['is_golden_correct'] == 'Incorrect')
                needs_rev = sum(1 for d in data if d['is_golden_correct'] == 'Needs_Revision')
                errors = sum(1 for d in data if d['is_golden_correct'] == 'Error')
                
                st.markdown("**Overall:**")
                st.metric("Total", total)
                st.metric("✅ Correct", f"{correct} ({correct/total*100:.1f}%)")
                st.metric("❌ Incorrect", f"{incorrect} ({incorrect/total*100:.1f}%)")
                st.metric("⚠️ Needs Revision", f"{needs_rev} ({needs_rev/total*100:.1f}%)")
                st.metric("⚡ Errors", errors)
                
                st.markdown("**By Dataset:**")
                datasets = self.get_unique_datasets(data)
                for dataset in datasets:
                    dataset_data = [d for d in data if self.get_dataset_type(d['question_id']) == dataset]
                    count = len(dataset_data)
                    st.text(f"{dataset}: {count} ({count/total*100:.1f}%)")
                
                st.markdown("---")
                
                # Session progress
                st.subheader("📝 Session Progress")
                st.metric("Decisions Made", len(st.session_state.decisions_made))
                st.metric("Changes Applied", len(st.session_state.changes_made))
                st.metric("Accepted (No Change)", 
                         sum(1 for d in st.session_state.decisions_made.values() 
                             if d['decision_type'] == 'accepted'))
                st.metric("Skipped", 
                         sum(1 for d in st.session_state.decisions_made.values() 
                             if d['decision_type'] == 'skipped'))
                
                st.markdown("---")
                st.info(f"💾 **Files saved to:**\n- Corrected QA: `qa_pairs_corrected/`\n- Logs: `verification_logs/`")
                
                # Checkpoint management
                st.markdown("---")
                st.subheader("💾 Checkpoints")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save Checkpoint", use_container_width=True):
                        self.save_checkpoint()
                        st.success("Checkpoint saved!")
                with col2:
                    if st.button("Generate Report", use_container_width=True):
                        summary = self.generate_summary_report()
                        st.success("Report generated!")
                        st.json(summary)
        
        # Main content
        if st.session_state.verification_data is None:
            st.info("👈 Click 'Load/Refresh Data' in the sidebar to begin")
            return
        
        filtered_data = self.filter_data(
            st.session_state.verification_data,
            st.session_state.filter_status,
            st.session_state.filter_dataset
        )
        
        if not filtered_data:
            st.warning(f"No items found with Status: {st.session_state.filter_status}, Dataset: {st.session_state.filter_dataset}")
            return
        
        # Navigation
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("⬅️ Previous", disabled=st.session_state.current_index == 0):
                st.session_state.current_index -= 1
                st.rerun()
        
        with col2:
            st.markdown(f"<h3 style='text-align: center;'>Item {st.session_state.current_index + 1} of {len(filtered_data)}</h3>", unsafe_allow_html=True)
        
        with col3:
            if st.button("Next ➡️", disabled=st.session_state.current_index >= len(filtered_data) - 1):
                st.session_state.current_index += 1
                st.rerun()
        
        st.markdown("---")
        
        # Get current item
        current_item = filtered_data[st.session_state.current_index]
        question_id = current_item['question_id']
        
        # Show if already processed
        if question_id in st.session_state.decisions_made:
            decision = st.session_state.decisions_made[question_id]
            st.info(f"ℹ️ Already processed: **{decision['decision_type'].upper()}** at {decision['timestamp']}")
        
        # Display verification status
        col1, col2 = st.columns([1, 3])
        with col1:
            self.display_status_badge(current_item['is_golden_correct'])
        with col2:
            st.metric("Confidence", current_item['confidence'])
        
        # Question details
        st.header("📋 Question Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Question ID:** {question_id}")
        with col2:
            st.info(f"**Question Type:** {current_item['question_type']}")
        with col3:
            dataset_type = self.get_dataset_type(question_id)
            st.info(f"**Dataset:** {dataset_type}")
        
        st.markdown(f"**Question:** {current_item['question']}")
        
        st.markdown("---")
        
        # Table display
        st.header("📊 Table Data")
        table_data = self.load_table_data(question_id)
        df = self.table_to_dataframe(table_data)
        
        if not df.empty:
            st.dataframe(df, use_container_width=True, height=300)
        else:
            st.warning("Table data not available")
        
        st.markdown("---")
        
        # Model responses
        st.header("🤖 Model Responses")
        for i, response in enumerate(current_item.get('model_responses', [])):
            with st.expander(f"Model {i+1}: {response.get('model_name', 'Unknown')}"):
                st.markdown(response.get('model_response', 'N/A'))
        
        st.markdown("---")
        
        # Verification analysis
        st.header("🔍 Verification Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Golden Answer")
            golden_answer = current_item['golden_answer']
            st.json(golden_answer)
        
        with col2:
            st.subheader("Suggested Corrected Answer")
            corrected_answer = current_item['corrected_answer']
            st.json(corrected_answer)
        
        # Reasoning
        st.subheader("💭 Reasoning")
        st.text_area(
            "Detailed Analysis",
            value=current_item['reasoning'],
            height=200,
            disabled=True,
            key=f"reasoning_{st.session_state.current_index}"
        )
        
        # Evidence summary
        st.subheader("📑 Evidence Summary")
        st.info(current_item['evidence_summary'])
        
        st.markdown("---")
        
        # Answer correction interface
        st.header("✏️ Correct Answer")
        
        current_answer = current_item['corrected_answer']
        if isinstance(current_answer, list):
            if len(current_answer) > 0 and isinstance(current_answer[0], list):
                answer_str = json.dumps(current_answer, indent=2)
            else:
                answer_str = json.dumps([current_answer], indent=2)
        else:
            answer_str = json.dumps([[str(current_answer)]], indent=2)
        
        st.info("💡 **Note:** Answers should be in nested list format: `[[\"answer\"]]` or `[[\"ans1\", \"ans2\"]]`")
        
        new_answer_str = st.text_area(
            "Edit the correct answer (nested JSON list format: [[\"answer\"]])",
            value=answer_str,
            height=150,
            key=f"answer_edit_{st.session_state.current_index}"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Correction", type="primary", use_container_width=True):
                try:
                    new_answer = json.loads(new_answer_str)
                    
                    if not isinstance(new_answer, list):
                        st.error("Answer must be a list")
                    elif len(new_answer) == 0:
                        st.error("Answer cannot be empty")
                    elif not isinstance(new_answer[0], list):
                        st.warning("Converting to nested list format: [[...]]")
                        new_answer = [new_answer]
                    
                    if self.update_qa_file(question_id, new_answer):
                        # Log the change
                        metadata = {
                            'dataset': dataset_type,
                            'question_type': current_item['question_type'],
                            'original_status': current_item['is_golden_correct'],
                            'confidence': current_item['confidence'],
                            'file': f"qa_pairs_corrected/{('_'.join(question_id.split('_')[:-1]))}_qa.json"
                        }
                        
                        self.log_change(question_id, current_item['golden_answer'], new_answer, metadata)
                        
                        # Log the decision
                        self.log_decision(question_id, 'changed', {
                            'old_answer': current_item['golden_answer'],
                            'new_answer': new_answer,
                            'metadata': metadata
                        })
                        
                        st.success("✅ Answer saved and logged!")
                        
                        if st.session_state.current_index < len(filtered_data) - 1:
                            st.session_state.current_index += 1
                            st.rerun()
                    else:
                        st.error("Failed to save correction")
                        
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON format: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with col2:
            if st.button("✅ Accept as Correct", use_container_width=True):
                # Log acceptance (no change needed)
                self.log_decision(question_id, 'accepted', {
                    'golden_answer': current_item['golden_answer'],
                    'status': current_item['is_golden_correct'],
                    'note': 'Golden answer is correct, no changes needed'
                })
                
                st.success("✅ Marked as correct (logged)")
                
                if st.session_state.current_index < len(filtered_data) - 1:
                    st.session_state.current_index += 1
                    st.rerun()
        
        with col3:
            if st.button("⏭️ Skip", use_container_width=True):
                # Log skip
                self.log_decision(question_id, 'skipped', {
                    'status': current_item['is_golden_correct'],
                    'note': 'Skipped for later review'
                })
                
                if st.session_state.current_index < len(filtered_data) - 1:
                    st.session_state.current_index += 1
                    st.rerun()
        
        # Export options
        st.markdown("---")
        st.header("📥 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Changes Log", use_container_width=True):
                st.success(f"Changes log: {self.changes_log_file}")
        
        with col2:
            if st.button("Export Decisions Log", use_container_width=True):
                st.success(f"Decisions log: {self.decisions_log_file}")
        
        with col3:
            if st.button("Export Summary Report", use_container_width=True):
                summary = self.generate_summary_report()
                st.success(f"Summary: {self.summary_log_file}")
                with st.expander("View Summary"):
                    st.json(summary)


def main():
    """Main entry point."""
    BASE_DIR = "data/processed/"
    
    if not Path(BASE_DIR).exists():
        st.error(f"Base directory not found: {BASE_DIR}")
        st.info("Please update BASE_DIR in the script to match your data location")
        return
    
    app = GoldenAnswerVerifier(BASE_DIR)
    app.run()


if __name__ == "__main__":
    main()