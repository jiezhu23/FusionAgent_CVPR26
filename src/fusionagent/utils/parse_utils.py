from collections import defaultdict
from typing import Any, Dict, List, Optional
import json
import re
from lxml import etree

# ANSI color codes for different conversation types
class Colors:
    USER = '\033[94m'      # Blue for user input
    ASSISTANT = '\033[92m'  # Green for model response
    TOOL_RESULT = '\033[95m' # Purple for tool_result
    RESET = '\033[0m'      # Reset color
    BOLD = '\033[1m'       # Bold text

class XMLLikeTagParser:
    """Lenient XML-like text parser that preserves top-level tag order.

    - Parses the provided string by wrapping it in a synthetic <root> to allow
      multiple top-level elements.
    - Uses lxml in recover mode to tolerate slightly malformed inputs.
    - Exposes utilities to retrieve ordered tags, per-tag contents, and to
      parse <tool_call> payloads into Python structures that can be used as
      **tool_kwargs.
    """

    def __init__(self, xml_like: str):
        self._xml_like: str = xml_like
        wrapped = f"<root>{xml_like}</root>"
        parser = etree.XMLParser(recover=True)
        self._root = etree.fromstring(wrapped.encode("utf-8"), parser=parser)

        # Only consider direct children of the synthetic root to preserve
        # the intended top-level sequence of tags.
        self._children: List[etree._Element] = list(self._root)

        # Map direct-children by tag for fast lookup; ordering is preserved in
        # self._children.
        self._tag_to_children: Dict[str, List[etree._Element]] = defaultdict(list)
        for child in self._children:
            self._tag_to_children[child.tag].append(child)

    def get_ordered_tags(self) -> List[str]:
        """Return tag names in the order they appear at the top level."""
        return [el.tag for el in self._children]

    def get_all_tags_in_order(self) -> List[str]:
        """Alias for get_ordered_tags to match naming expectations."""
        return self.get_ordered_tags()

    def get_first_content(self, tag: str) -> str:
        """Get text content of the first occurrence of a tag (top-level).

        Returns an empty string if the tag does not exist.
        """
        elements = self._tag_to_children.get(tag)
        if not elements:
            return ""
        return "".join(elements[0].itertext()).strip()

    def get_contents(self, tag: str) -> List[str]:
        """Get text contents for all occurrences of a given tag (top-level)."""
        return ["".join(el.itertext()).strip() for el in self._tag_to_children.get(tag, [])]

    def get_all_tag_contents(self) -> List[Dict[str, Any]]:
        """Return a list of dicts [{tag, content}] preserving top-level order."""
        results: List[Dict[str, Any]] = []
        for el in self._children:
            results.append({
                "tag": el.tag,
                "content": "".join(el.itertext()).strip(),
            })
        return results

    def parse_tool_calls(self) -> List[Dict[str, Any]]:
        """Parse all <tool_call> elements.

        Expected content is a JSON object like:
        {"name": "tool_name", "parameters": {"arg1": 1, ...}}

        Returns a list of objects: {"name": str, "tool_kwargs": dict}.
        Catches parsing errors and returns them as part of the list, with an 'error' key.
        """
        tool_calls: List[Dict[str, Any]] = []
        for el in self._tag_to_children.get("tool_call", []):
            raw = "".join(el.itertext()).strip()
            if not raw:
                # Skip empty <tool_call> tags.
                continue

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as err:
                print(f"Invalid JSON inside <tool_call>: {err}")
                tool_calls.append({"error": f"Invalid JSON: {err}", "raw_content": raw})
                continue
            
            if isinstance(payload, list):
                if len(payload) == 1 and isinstance(payload[0], dict):
                    payload = payload[0]
                else:
                    print(f"Invalid <tool_call> payload: expected a single JSON object, but got a list: {raw}")
                    tool_calls.append({"error": "Invalid <tool_call> payload: unexpected list", "raw_content": raw})
                    continue

            if not isinstance(payload, dict):
                print(f"Invalid <tool_call> payload: must be a JSON object: {raw}")
                tool_calls.append({"error": "Invalid <tool_call> payload: not an object", "raw_content": raw})
                continue

            name = payload.get("name")
            if not isinstance(name, str) or not name:
                print(f"<tool_call> payload must include a non-empty 'name' string: {raw}")
                tool_calls.append({"error": f"Invalid <tool_call> payload: missing 'name'", "raw_content": raw})
                continue

            # Accept common keys for parameters; normalize to tool_kwargs
            params: Optional[Dict[str, Any]] = None
            for key in ("parameters", "kwargs", "tool_kwargs"):
                if key in payload:
                    params = payload[key]
                    break

            tool_kwargs: Dict[str, Any] = {}
            if params is not None:
                if not isinstance(params, dict):
                    print(f"'parameters' (or 'kwargs'/'tool_kwargs') must be an object, but got {type(params)}")
                    tool_calls.append({
                        "error": f"'parameters' must be an object, but got {type(params).__name__}",
                        "name": name,
                    })
                    continue
                tool_kwargs = params

            tool_calls.append({
                "name": name,
                "tool_kwargs": tool_kwargs,
            })
        return tool_calls


class ConversationParser:
    """Efficient parser for multi-turn conversations with XML-like tags.
    
    This class is designed to parse conversations containing assistant responses
    with <think>, <tool_call>, <answer>, and <tool_result> tags.
    """
    
    def __init__(self, conversation: List[dict]):
        """Initialize the parser with a conversation list.
        
        Args:
            conversation: List of conversation turns, each containing 'role' and 'content'
        """
        self.conversation = conversation
        self.assistant_turns = []
        self.parsed_turns = []
        
        # Extract assistant turns and parse them
        for turn in conversation:
            if turn.get('role') == 'assistant':
                content = turn.get('content', '')
                if content.strip():
                    self.assistant_turns.append(content)
                    try:
                        parser = XMLLikeTagParser(content)
                        self.parsed_turns.append(parser)
                    except Exception as e:
                        # If parsing fails, create empty parser
                        print(f"Warning: Failed to parse turn content: {e}")
                        self.parsed_turns.append(None)
    
    def get_total_turns(self) -> int:
        """Get the total number of assistant turns."""
        return len(self.assistant_turns)
    
    def has_complete_format(self, turn_idx: int) -> bool:
        """Check if a specific turn has complete format.
        
        Complete format means the turn content matches either:
        1. <think>...</think> followed by <tool_call>...</tool_call>
        2. <think>...</think> followed by <answer>...</answer>
        
        Args:
            turn_idx: Index of the turn to check
            
        Returns:
            True if format is complete, False otherwise
        """
        pattern_tool_call = r"<think>.*?</think>\s*<tool_call>.*?</tool_call>"
        pattern_answer = r"<think>.*?</think>\s*<answer>.*?</answer>"

        if turn_idx >= len(self.assistant_turns):
            return False
            
        turn_content = self.assistant_turns[turn_idx]
        
        # Check if the turn content matches either pattern
        match_tool_call = re.search(pattern_tool_call, turn_content, re.DOTALL)
        match_answer = re.search(pattern_answer, turn_content, re.DOTALL)
        
        return match_tool_call is not None or match_answer is not None
    
    def has_complete_format_v2(self, turn_idx: int) -> bool:
        """Check if a specific turn has a strict complete format.
        
        Complete format means the turn content matches either:
        1. <think>...</think> followed by <tool_call>...</tool_call>
        2. <think>...</think> followed by <answer>...</answer>
        and no other tags are present at the top level. All tags must be properly closed.
        
        Args:
            turn_idx: Index of the turn to check
            
        Returns:
            True if format is strictly complete, False otherwise
        """
        if turn_idx >= len(self.assistant_turns):
            return False

        # First, check for well-formedness (all tags must be closed)
        turn_content = self.assistant_turns[turn_idx]
        # lxml requires a single root element
        wrapped_content = f"<root>{turn_content}</root>"
        try:
            # Use a strict parser (default) to check for well-formedness
            etree.fromstring(wrapped_content.encode("utf-8"))
        except etree.XMLSyntaxError:
            return False  # Malformed XML (e.g., unclosed tags)

        # Then, check the tag order using the existing leniently parsed result
        if turn_idx >= len(self.parsed_turns) or self.parsed_turns[turn_idx] is None:
            return False
            
        parser = self.parsed_turns[turn_idx]
        ordered_tags = parser.get_ordered_tags()
        
        is_tool_call_format = (ordered_tags == ['think', 'tool_call'])
        is_answer_format = (ordered_tags == ['think', 'answer'])
        
        return is_tool_call_format or is_answer_format
    
    def has_complete_format_fast(self, turn_idx: int) -> bool:
        """Check if a specific turn has a strict complete format for fast answering.
        
        Complete format means the turn content matches either:
        1. <tool_call>...</tool_call>
        2. <answer>...</answer>
        and no other tags are present at the top level. All tags must be properly closed.
        
        Args:
            turn_idx: Index of the turn to check
            
        Returns:
            True if format is strictly complete, False otherwise
        """
        if turn_idx >= len(self.assistant_turns):
            return False

        # First, check for well-formedness (all tags must be closed)
        turn_content = self.assistant_turns[turn_idx]
        # lxml requires a single root element
        wrapped_content = f"<root>{turn_content}</root>"
        try:
            # Use a strict parser (default) to check for well-formedness
            etree.fromstring(wrapped_content.encode("utf-8"))
        except etree.XMLSyntaxError:
            return False  # Malformed XML (e.g., unclosed tags)

        # Then, check the tag order using the existing leniently parsed result
        if turn_idx >= len(self.parsed_turns) or self.parsed_turns[turn_idx] is None:
            return False
            
        parser = self.parsed_turns[turn_idx]
        ordered_tags = parser.get_ordered_tags()
        
        is_tool_call_format = (ordered_tags == ['tool_call'])
        is_answer_format = (ordered_tags == ['answer'])
        
        return is_tool_call_format or is_answer_format
    
    def has_complete_format_fast_v2(self, turn_idx: int) -> bool:
        """Check if a specific turn has a strict complete format for fast answering.
        
        Complete format means the turn content matches either:
        1. <tool_call>...</tool_call>
        2. <answer>...</answer>
        and no other tags are present at the top level. All tags must be properly closed.
        
        Args:
            turn_idx: Index of the turn to check
            
        Returns:
            True if format is strictly complete, False otherwise
        """
        if turn_idx >= len(self.assistant_turns):
            return False

        # First, check for well-formedness (all tags must be closed)
        turn_content = self.assistant_turns[turn_idx]
        # lxml requires a single root element
        wrapped_content = f"<root>{turn_content}</root>"
        try:
            # Use a strict parser (default) to check for well-formedness
            etree.fromstring(wrapped_content.encode("utf-8"))
        except etree.XMLSyntaxError:
            return False  # Malformed XML (e.g., unclosed tags)

        # Then, check the tag order using the existing leniently parsed result
        if turn_idx >= len(self.parsed_turns) or self.parsed_turns[turn_idx] is None:
            return False
            
        parser = self.parsed_turns[turn_idx]
        ordered_tags = parser.get_ordered_tags()
        
        is_tool_call_format = (ordered_tags == ['tool_call'])
        is_answer_format = (ordered_tags == ['answer'])
        
        return is_tool_call_format or is_answer_format
    
    def get_tool_success_rate(self) -> float:
        """Extract all tool_result contents from assistant responses.
        Returns:
            List of parsed tool_result contents as dictionaries
        """
        tool_call_count = 0
        tool_call_success_count = 0
        # skip the first two turns (system and user)
        for turn in self.conversation[2:]:
            if turn.get('role') == 'system':
                content = turn.get('content', '')
                if '<tool_result>' in content and '</tool_result>' in content:
                    # Extract content between tool_result tags
                    start = content.find('<tool_result>') + len('<tool_result>')
                    end = content.find('</tool_result>')
                    result_content = content[start:end].strip()
                    
                    # Try to parse as dictionary
                    result_data = eval(result_content)
                    tool_call_count += 1
                    try:
                        if result_data.get('error') is None:
                            tool_call_success_count += 1
                    except Exception as e:
                        print(f"Warning: Failed to parse tool_result: {e}, result_content: {result_content}")
                        continue
        
        return tool_call_success_count / tool_call_count if tool_call_count > 0 else 0.0
    
    def __str__(self) -> str:
        """Custom string representation showing conversation content with color coding.
        Returns a formatted string with each turn's role and content, using different colors for:
        - User input (blue)
        - Model response (green) 
        - Tool result (yellow)
        """
        if not self.conversation:
            return "ConversationParser(empty conversation)"
        
        lines = []
        for i, turn in enumerate(self.conversation[1:]):
            turn_content = turn.get('content', '')
            role = turn.get('role', '')
            
            if isinstance(turn_content, list):
                turn_content = f"{turn_content[1]['text']} ({','.join(turn_content[0]['video'])})"
            
            # Apply color coding based on content type
            colored_content = self._apply_color_coding(turn_content, role)
            lines.append(colored_content)
        
        return "\n".join(lines)
    
    def _apply_color_coding(self, content: str, role: str) -> str:
        """Apply color coding to content based on its type.
        
        Args:
            content: The content to color
            role: The role of the speaker (user, assistant, system)
            
        Returns:
            Content with ANSI color codes applied
        """
        if role == 'user':
            return f"{Colors.USER}{content}{Colors.RESET}"
        elif role == 'system' and '<tool_result>' in content:
            return f"{Colors.TOOL_RESULT}{content}{Colors.RESET}"
        elif role == 'assistant':
            return f"{Colors.ASSISTANT}{content}{Colors.RESET}"
        else:
            return content
    
    def _color_assistant_content(self, content: str) -> str:
        """Apply detailed color coding to assistant content.
        
        Colors different parts of assistant responses:
        - <think> tags and content
        - <tool_call> tags and content  
        - <answer> tags and content
        - Plain text responses
        
        Args:
            content: Assistant content to color
            
        Returns:
            Content with appropriate color coding
        """
        # If content contains XML-like tags, apply granular coloring
        if '<think>' in content or '<tool_call>' in content or '<answer>' in content:
            # Color <think> sections
            content = re.sub(r'(<think>.*?</think>)', 
                           rf'{Colors.ASSISTANT}\1{Colors.RESET}', 
                           content, flags=re.DOTALL)
            
            # Color <tool_call> sections  
            content = re.sub(r'(<tool_call>.*?</tool_call>)', 
                           rf'{Colors.BOLD}{Colors.ASSISTANT}\1{Colors.RESET}', 
                           content, flags=re.DOTALL)
            
            # Color <answer> sections
            content = re.sub(r'(<answer>.*?</answer>)', 
                           rf'{Colors.BOLD}{Colors.ASSISTANT}\1{Colors.RESET}', 
                           content, flags=re.DOTALL)
            
            return content
        else:
            # Plain assistant response
            return f"{Colors.ASSISTANT}{content}{Colors.RESET}"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"ConversationParser(conversation={len(self.conversation)} turns, assistant_turns={len(self.assistant_turns)})"


def get_all_tags_in_order(xml_like: str) -> List[str]:
    """Convenience wrapper to get top-level tags in order of appearance."""
    parser = XMLLikeTagParser(xml_like)
    return parser.get_all_tags_in_order()


def parse_tool_call_objects(xml_like: str) -> List[Dict[str, Any]]:
    """Convenience wrapper to parse <tool_call> elements into call specs."""
    parser = XMLLikeTagParser(xml_like)
    return parser.parse_tool_calls()
