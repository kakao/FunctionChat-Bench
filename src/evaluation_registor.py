from functools import wraps
from src import formatter


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

    def get_eval_output_length(self):
        """
        Returns the number of evaluation outputs stored in the list.

        Returns:
            int: The length of the evaluation output list.
        """
        return len(self.eval_output)

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

    def display(self):
        for data in self.eval_output:
            is_pass = formatter.convert_eval_key(data['evaluate_response'])
            self.add_eval_dic(type_of_output=data['model_request']['type_of_output'],
                              is_pass=is_pass, serial_num=data['model_request']['serial_num'])
            self.add_eval_dic_per_category(category=data['model_request']['category'],
                                           is_pass=is_pass, serial_num=data['model_request']['serial_num'])
        print("Pass Count")
        total_cnt = 0
        tot_pass_cnt_per_cate = 0
        categories = sorted(set(self.eval_dic_per_category.keys()))
        for category in categories:
            if category in self.eval_dic_per_category:
                pass_cnt = len(self.eval_dic_per_category[category].get('pass', []))
                tot_pass_cnt_per_cate += pass_cnt
                case_tot_cnt_per_cate = pass_cnt + len(self.eval_dic_per_category[category].get('fail', []))
                total_cnt += case_tot_cnt_per_cate
                print(f"  {category} : {pass_cnt}/{case_tot_cnt_per_cate}")
        print(f"  total : {tot_pass_cnt_per_cate}/{total_cnt}")
        total_cnt = 0
        tot_pass_cnt_per_cate = 0
        print("Pass Rate")
        for category in categories:
            if category in self.eval_dic_per_category:
                pass_cnt = len(self.eval_dic_per_category[category].get('pass', []))
                fail_cnt = len(self.eval_dic_per_category[category].get('fail', []))
                tot_pass_cnt_per_cate += pass_cnt
                case_tot_cnt_per_cate = pass_cnt + fail_cnt
                total_cnt += case_tot_cnt_per_cate
                print(f"  {category} : {pass_cnt/case_tot_cnt_per_cate:.2f}")
        print(f"  total : {tot_pass_cnt_per_cate/total_cnt:.2f}")
        total_pass_count = 0


class DialogEvaluationRegistor(AbstractEvaluationRegistor):
    def __init__(self):
        super().__init__()
        self.max_size = 200
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

    def display(self):
        # set eval_dic
        for data in self.eval_output:
            inp = data['model_request']
            serial_num = inp['serial_num']
            type_of_output = inp['type_of_output']
            is_pass = formatter.convert_eval_key(data['evaluate_response'])
            self.add_eval_dic(type_of_output=type_of_output,
                              is_pass=is_pass, serial_num=serial_num)
        tot_pass_cnt = 0
        print("\n* pass count")
        for type_of_output in self.types_of_output:
            if type_of_output in self.eval_dic:
                pass_cnt = len(self.eval_dic[type_of_output].get('pass', []))
                tot_pass_cnt += pass_cnt
                case_tot_cnt = pass_cnt + len(self.eval_dic[type_of_output].get('fail', []))
                print(f"  {type_of_output} : {pass_cnt}/{case_tot_cnt}")
        print(f"  total : {tot_pass_cnt}/{self.max_size}")
        #
        print("\n* pass rate")
        for type_of_output in self.types_of_output:
            if type_of_output in self.eval_dic:
                pass_cnt = len(self.eval_dic[type_of_output].get('pass', []))
                case_tot_cnt = pass_cnt + len(self.eval_dic[type_of_output].get('fail', []))
                print(f"  {type_of_output} : {pass_cnt/case_tot_cnt:.2f}")
        print(f" avg(micro) : {tot_pass_cnt/self.max_size}")


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

    def display(self):
        # set eval_dic
        for data in self.eval_output:
            inp = data['model_request']
            tools_type = inp['tools_type']
            serial_num = inp['serial_num']
            is_pass = formatter.convert_eval_key(data['evaluate_response'])
            self.add_eval_dic(tools_type=tools_type,
                              is_pass=is_pass, serial_num=serial_num)
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
