from functools import wraps
from src import formatter
from src.constants import PASS_STR, FAIL_STR, MAX_DIALOG_EVAL_SIZE


def validate_params(required_keys):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            missing_keys = [key for key in required_keys if key not in kwargs or kwargs[key] is None]
            if missing_keys:
                print(kwargs)
                raise ValueError(f"Missing required parameters: {', '.join(missing_keys)}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


class AbstractEvaluationRegistor:
    """
    An abstract base class for evaluation registers, designed to handle and store evaluation results.
    This class provides a template for creating specific evaluation register classes that implement
    customized display and additional data handling functionalities.
    """
    def __init__(self):
        self.eval_dic = {}
        self.eval_output = []
        self.indexing_dic = {}

    def get_eval_output_length(self):
        """
        Returns the number of evaluation outputs stored in the list.

        Returns:
            int: The length of the evaluation output list.
        """
        return len(self.eval_output)

    def get_pass_count(self):
        """
        Returns the total number of passing evaluations.

        Returns:
            int: The total number of passing evaluations.
        """
        pass_count = 0
        for data in self.eval_output:
            if formatter.convert_eval_key(data['evaluate_response']) == PASS_STR:
                pass_count += 1
        return pass_count

    def get_pass_ratio(self):
        """
        Returns the ratio of passing evaluations to total evaluations.

        Returns:
            float: The ratio of passing evaluations (0.0 to 1.0).
        """
        total = len(self.eval_output)
        if total == 0:
            return 0.0
        return self.get_pass_count() / total

    def get_detailed_scores(self):
        """
        Returns detailed scores for each evaluation category.

        Returns:
            dict: A dictionary containing detailed scores for each category.
        """
        scores = {}
        for data in self.eval_output:
            category = data['model_request'].get('category', 'unknown')
            if category not in scores:
                scores[category] = {PASS_STR: 0, FAIL_STR: 0}
            is_pass = formatter.convert_eval_key(data['evaluate_response'])
            scores[category][is_pass] += 1
        return scores

    def set_eval_output(self, eval_output):
        """
        Sets the evaluation output list to a new list of outputs.

        Parameters:
            eval_output (list): A list of evaluation outputs to replace the existing list.
        """
        self.eval_output = eval_output

    def add_eval_output(self, output):
        """
        Adds a single evaluation output to the end of the evaluation output list.

        Parameters:
            output (any): An evaluation result to be added to the list.
        """
        self.eval_output.append(output)

    def add_eval_dic(self, **kwargs):
        """
        Abstract method to add additional evaluation data to the eval_dic dictionary.
        This method must be implemented by subclasses.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def display(self):
        """
        Abstract method to display or report the evaluation results.
        This method must be implemented by subclasses to define how evaluation results are presented.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class CommonEvaluationRegistor(AbstractEvaluationRegistor):
    def __init__(self):
        super().__init__()
        self.types_of_output = ['call', 'completion', 'slot', 'relevance']
        self.eval_dic_per_category = {}

    @validate_params(['type_of_output', 'is_pass', 'serial_num'])
    def add_eval_dic(self, **kwargs):
        type_of_output = kwargs.get('type_of_output')
        is_pass = kwargs.get('is_pass')
        serial_num = kwargs.get('serial_num')
        if type_of_output not in self.eval_dic:
            self.eval_dic[type_of_output] = {}
        if is_pass not in self.eval_dic[type_of_output]:
            self.eval_dic[type_of_output][is_pass] = []
        self.eval_dic[type_of_output][is_pass].append(serial_num)
        self.indexing_dic[serial_num] = (type_of_output, is_pass)

    @validate_params(['category', 'is_pass', 'serial_num'])
    def add_eval_dic_per_category(self, **kwargs):
        category = kwargs.get('category')
        is_pass = kwargs.get('is_pass')
        serial_num = kwargs.get('serial_num')
        if category not in self.eval_dic_per_category:
            self.eval_dic_per_category[category] = {}
        if is_pass not in self.eval_dic_per_category[category]:
            self.eval_dic_per_category[category][is_pass] = []
        self.eval_dic_per_category[category][is_pass].append(serial_num)

    def set_eval_dic(self):
        for data in self.eval_output:
            is_pass = formatter.convert_eval_key(data['evaluate_response'])
            self.add_eval_dic(type_of_output=data['model_request']['type_of_output'],
                              is_pass=is_pass, serial_num=data['model_request']['serial_num'])
            self.add_eval_dic_per_category(category=data['model_request']['category'],
                                           is_pass=is_pass, serial_num=data['model_request']['serial_num'])

    def display(self):
        self.set_eval_dic()
        print("Pass Count")
        total_cnt = 0
        tot_pass_cnt_per_cate = 0
        categories = sorted(set(self.eval_dic_per_category.keys()))
        for category in categories:
            if category in self.eval_dic_per_category:
                pass_cnt = len(self.eval_dic_per_category[category].get(PASS_STR, []))
                tot_pass_cnt_per_cate += pass_cnt
                case_tot_cnt_per_cate = pass_cnt + len(self.eval_dic_per_category[category].get(FAIL_STR, []))
                total_cnt += case_tot_cnt_per_cate
                print(f"  {category} : {pass_cnt}/{case_tot_cnt_per_cate}")
        print(f"  total : {tot_pass_cnt_per_cate}/{total_cnt}")
        total_cnt = 0
        tot_pass_cnt_per_cate = 0
        print("Pass Rate")
        for category in categories:
            if category in self.eval_dic_per_category:
                pass_cnt = len(self.eval_dic_per_category[category].get(PASS_STR, []))
                fail_cnt = len(self.eval_dic_per_category[category].get(FAIL_STR, []))
                tot_pass_cnt_per_cate += pass_cnt
                case_tot_cnt_per_cate = pass_cnt + fail_cnt
                total_cnt += case_tot_cnt_per_cate
                if case_tot_cnt_per_cate > 0:
                    print(f"  {category} : {pass_cnt/case_tot_cnt_per_cate:.2f}")
                else:
                    print(f"  {category} : 0.00")
        if total_cnt > 0:
            print(f"  total : {tot_pass_cnt_per_cate/total_cnt:.2f}")
        else:
            print(f"  total : 0.00")

    def get_score(self):
        score_dict = {}
        if len(self.eval_dic) == 0:
            self.set_eval_dic()
        total_cnt = 0
        tot_pass_cnt_per_cate = 0
        categories = sorted(set(self.eval_dic_per_category.keys()))
        for category in categories:
            if category in self.eval_dic_per_category:
                pass_cnt = len(self.eval_dic_per_category[category].get(PASS_STR, []))
                tot_pass_cnt_per_cate += pass_cnt
                case_tot_cnt_per_cate = pass_cnt + len(self.eval_dic_per_category[category].get(FAIL_STR, []))
                total_cnt += case_tot_cnt_per_cate
                score_dict[f'{category} pass cnt'] = pass_cnt
                score_dict[f'{category} pass rate'] = pass_cnt/case_tot_cnt_per_cate if case_tot_cnt_per_cate > 0 else 0.00
        score_dict['total_pass_cnt'] = tot_pass_cnt_per_cate
        score_dict['total_cnt'] = total_cnt
        score_dict['total_pass_rate'] = tot_pass_cnt_per_cate/total_cnt
        return score_dict

class DialogEvaluationRegistor(AbstractEvaluationRegistor):
    def __init__(self):
        super().__init__()
        self.max_size = MAX_DIALOG_EVAL_SIZE
        self.types_of_output = ['call', 'completion', 'slot', 'relevance']

    @validate_params(['type_of_output', 'is_pass', 'serial_num'])
    def add_eval_dic(self, **kwargs):
        type_of_output = kwargs.get('type_of_output')
        is_pass = kwargs.get('is_pass')
        serial_num = kwargs.get('serial_num')
        if type_of_output not in self.eval_dic:
            self.eval_dic[type_of_output] = {}
        if is_pass not in self.eval_dic[type_of_output]:
            self.eval_dic[type_of_output][is_pass] = []
        self.eval_dic[type_of_output][is_pass].append(serial_num)
        self.indexing_dic[serial_num] = (type_of_output, is_pass)

    def set_eval_dic(self):
        # set eval_dic
        for data in self.eval_output:
            inp = data['model_request']
            serial_num = inp['serial_num']
            if serial_num in self.indexing_dic:
                continue
            type_of_output = inp['type_of_output']
            is_pass = formatter.convert_eval_key(data['evaluate_response'])
            self.add_eval_dic(type_of_output=type_of_output,
                              is_pass=is_pass, serial_num=serial_num)

    def display(self):
        self.set_eval_dic()

        tot_pass_cnt = 0
        print("\n* pass count")
        for type_of_output in self.types_of_output:
            if type_of_output in self.eval_dic:
                pass_cnt = len(self.eval_dic[type_of_output].get(PASS_STR, []))
                tot_pass_cnt += pass_cnt
                case_tot_cnt = pass_cnt + len(self.eval_dic[type_of_output].get(FAIL_STR, []))
                print(f"  {type_of_output} : {pass_cnt}/{case_tot_cnt}")
        print(f"  total : {tot_pass_cnt}/{self.max_size}")
        #
        print("\n* pass rate")
        for type_of_output in self.types_of_output:
            if type_of_output in self.eval_dic:
                pass_cnt = len(self.eval_dic[type_of_output].get(PASS_STR, []))
                case_tot_cnt = pass_cnt + len(self.eval_dic[type_of_output].get(FAIL_STR, []))
                print(f"  {type_of_output} : {pass_cnt/case_tot_cnt:.2f}")
        print(f" avg(micro) : {tot_pass_cnt/self.max_size}")

    def get_score(self):
        if len(self.eval_dic) == 0:
            self.set_eval_dic()

        score_dict = {}
        tot_pass_cnt = 0
        for type_of_output in self.types_of_output:
            if type_of_output in self.eval_dic:
                pass_cnt = len(self.eval_dic[type_of_output].get(PASS_STR, []))
                tot_pass_cnt += pass_cnt
                case_tot_cnt = pass_cnt + len(self.eval_dic[type_of_output].get(FAIL_STR, []))
                score_dict[f'{type_of_output} pass cnt'] = pass_cnt
                score_dict[f'{type_of_output} pass rate'] = pass_cnt/case_tot_cnt
        score_dict['total_pass_cnt'] = tot_pass_cnt
        score_dict['total_cnt'] = self.max_size
        score_dict['avg(micro)'] = tot_pass_cnt/self.max_size
        return score_dict    


class SingleCallEvaluationRegistor(AbstractEvaluationRegistor):
    def __init__(self):
        super().__init__()
        self.eval_dic_of_tools_type = {}

    @validate_params(['is_pass', 'serial_num', 'tools_type'])
    def add_eval_dic(self, **kwargs):
        is_pass = kwargs.get('is_pass')
        serial_num = kwargs.get('serial_num')
        tools_type = kwargs.get('tools_type')
        if is_pass not in self.eval_dic:
            self.eval_dic[is_pass] = []
        self.eval_dic[is_pass].append(serial_num)
        ##
        if tools_type not in self.eval_dic_of_tools_type:
            self.eval_dic_of_tools_type[tools_type] = {}
        if is_pass not in self.eval_dic_of_tools_type[tools_type]:
            self.eval_dic_of_tools_type[tools_type][is_pass] = []
        self.eval_dic_of_tools_type[tools_type][is_pass].append(serial_num)
        key = f'{serial_num}-{tools_type}'
        self.indexing_dic[key] = is_pass

    def set_eval_dic(self):
        for data in self.eval_output:
            inp = data['model_request']
            serial_num = inp['serial_num']
            tools_type = inp['tools_type']
            key = f'{serial_num}-{tools_type}'
            if key in self.indexing_dic:
                continue
            is_pass = formatter.convert_eval_key(data['evaluate_response'])
            self.add_eval_dic(tools_type=tools_type,
                              is_pass=is_pass, serial_num=serial_num)

    def display(self):
        self.set_eval_dic()
        tot_cnt = 0
        for tools_type, values in self.eval_dic_of_tools_type.items():
            total_count = 0
            if values:
                for is_pass, serial_num_list in self.eval_dic_of_tools_type[tools_type].items():
                    total_count += len(serial_num_list)
                    tot_cnt += len(serial_num_list)
                print(f'[[{tools_type} TOTAL {total_count}]]')
                for is_pass, serial_num_list in self.eval_dic_of_tools_type[tools_type].items():
                    print(f'* {is_pass} : {len(serial_num_list)}')
                print()
        print()
        print(f"[[TOTAL {tot_cnt}]]")
        for is_pass, serial_num_list in self.eval_dic.items():
            print(f"{is_pass}\t{len(serial_num_list)}")

    def get_score(self):
        score_dict = {}
        if len(self.eval_dic) == 0:
            self.set_eval_dic()
        tot_cnt = 0
        tot_pass_cnt = 0
        for tools_type, values in self.eval_dic_of_tools_type.items():
            tools_type_total_count = 0
            if values:
                for is_pass, serial_num_list in self.eval_dic_of_tools_type[tools_type].items():
                    tools_type_total_count += len(serial_num_list)
                    tot_cnt += len(serial_num_list)
                score_dict[f'{tools_type} total'] = tools_type_total_count
                for is_pass, serial_num_list in self.eval_dic_of_tools_type[tools_type].items():
                    if is_pass == PASS_STR:
                        tot_pass_cnt += len(serial_num_list)
                        score_dict[f'{tools_type} pass cnt'] = len(serial_num_list)
                        score_dict[f'{tools_type} pass rate'] = len(serial_num_list)/tools_type_total_count
        score_dict['total_cnt'] = tot_cnt
        score_dict['total_pass_cnt'] = tot_pass_cnt
        score_dict['total_pass_rate'] = tot_pass_cnt/tot_cnt
        return score_dict