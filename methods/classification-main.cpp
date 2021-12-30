#include "classification-main.h"

using namespace std;

extern TupleInfo tuple_info_sum;
extern uint64_t check_hash_cost;
extern uint64_t check_group_cost;
extern uint64_t check_rule_cost;

void PrintProgramState(CommandStruct command, ProgramState *program_state) {

    if (command.print_mode == 0) {
        printf("\n");
        // PrintTree(program_state);
        printf("method_name: %s\n\n", command.method_name.c_str());
        printf("prefix_dims_num: %d\n\n", command.prefix_dims_num);

        printf("rules_num: %d\n", program_state->rules_num);
        printf("traces_num: %d\n", program_state->traces_num);
        printf("data_memory_size: %.3f MB\n", program_state->data_memory_size);
        printf("index_memory_size: %.3f MB\n\n", program_state->index_memory_size);

        printf("build_time: %.2f S\n", program_state->build_time);
        printf("lookup_speed: %.2f MLPS\n", program_state->lookup_speed);
        printf("insert_speed: %.2f MUPS\n", program_state->insert_speed);
        printf("delete_speed: %.2f MUPS\n", program_state->delete_speed);
        printf("update_speed: %.2f MUPS\n\n", program_state->update_speed);

        printf("tuples_num: %d\n", program_state->tuples_num);
        printf("tuples_sum: %d\n", program_state->tuples_sum);
        printf("access_tuples_sum: %d\taccess_tuples_avg: %.2f\taccess_tuples_max: %d\n",
            program_state->access_tuples.sum,
            1.0 * program_state->access_tuples.sum / program_state->traces_num,
            program_state->access_tuples.maxn);
        printf("access_node_sum: %d\taccess_node_avg: %.2f\taccess_node_max: %d\n",
            program_state->access_nodes.sum,
            1.0 * program_state->access_nodes.sum / program_state->traces_num,
            program_state->access_nodes.maxn);
        printf("access_rules_sum: %d\taccess_rules_avg: %.2f\taccess_rules_max: %d\n",
            program_state->access_rules.sum,
            1.0 * program_state->access_rules.sum / program_state->traces_num,
            program_state->access_rules.maxn);
        
        printf("next_layer_num: %d\n", program_state->next_layer_num);
        printf("\n\n");
        
        if (command.method_name == "DynamicTuple_Demo") {
            printf("DynamicTuple_Demo access and predicted access:\n\n");
            printf("Part 1:\n");
            printf("access_tuples: %d\tpredict: %ld\n\n", program_state->access_tuples.sum, tuple_info_sum.check_tuple_num);

            printf("Part 2:\n");
            int matching_access = program_state->low_priority_matching_access + program_state->high_priority_matching_access;
            printf("group_matching_access: %d\n", matching_access);
            printf("low_priority_matching_access: %d\tpredict:%ld\n", program_state->low_priority_matching_access, tuple_info_sum.match_group_num);
            printf("high_priority_matching_access: %d\n\n", program_state->high_priority_matching_access);

            printf("Part 3:\n");
            int collision_access = program_state->low_priority_collision_access + program_state->high_priority_collision_access;
            printf("group_collision_access: %d\n", collision_access);
            printf("low_priority_collision_access: %d\tpredict: %ld\n", program_state->low_priority_collision_access, tuple_info_sum.collide_group_num);
            printf("high_priority_collision_access: %d\n\n", program_state->high_priority_collision_access);

            printf("Part 4:\n");
            printf("access_rules: %d\n", program_state->access_rules.sum);
            printf("low_priority_rule_access: %d\tpredict: %ld\n", program_state->low_priority_rule_access, tuple_info_sum.check_rule_num);
            printf("high_priority_rule_access: %d\n\n", program_state->high_priority_rule_access);

            printf("Summary:\n");
            uint64_t cost = program_state->access_tuples.sum * check_hash_cost + 
                            (matching_access + collision_access) * check_group_cost +
                            program_state->access_rules.sum * check_rule_cost;
            uint64_t predict_cost = tuple_info_sum.check_tuple_num * check_hash_cost + 
                                    (tuple_info_sum.match_group_num + tuple_info_sum.collide_group_num) * check_group_cost +
                                    tuple_info_sum.check_rule_num* check_rule_cost;
            printf("cost: %ld\tpredict: %ld\n", cost, predict_cost);
            printf("error ratio %.4f\n", 1.0 * (max(cost, predict_cost) - min(cost, predict_cost)) / cost);
            printf("\n\n");
        }
    } else if (command.print_mode == 1) {
        //printf("%s\t", command.method_name.c_str());
        printf("%d\t", program_state->rules_num);
        printf("%d\t", program_state->traces_num);

        printf("%.3f\t", program_state->data_memory_size);
        printf("%.3f\t", program_state->index_memory_size);

        printf("%.4f\t", program_state->build_time);
        printf("%.4f\t", program_state->lookup_speed);
        printf("%.2f\t", program_state->insert_speed);
        printf("%.2f\t", program_state->delete_speed);
        printf("%.2f\t", program_state->update_speed);

        printf("%d\t", program_state->tuples_num);
        printf("%d\t", program_state->tuples_sum);
        printf("%d\t%.2f\t%d\t", program_state->access_tuples.sum, 1.0 * program_state->access_tuples.sum / program_state->traces_num, program_state->access_tuples.maxn);
        printf("%d\t%.2f\t%d\t", program_state->access_rules.sum, 1.0 * program_state->access_rules.sum / program_state->traces_num, program_state->access_rules.maxn);

        printf("%d\t", program_state->next_layer_num);
        printf("\n");
    }
}

int ClassificationMain(CommandStruct command) {

    vector<Rule*> rules = ReadRules(command.rules_file, command.rules_shuffle);

    vector<Trace*> traces = ReadTraces(command.traces_file);

    if (command.prefix_dims_num == 5)
        rules = RulesPortPrefix(rules, true);
    
    vector<int> ans = GenerateAns(rules, traces, command.force_test);
    
    ProgramState *program_state = new ProgramState();
    program_state->rules_num = rules.size();
    program_state->traces_num = traces.size();

    if (command.method_name == "MultilayerTuple" || command.method_name == "DimTSS" || 
        command.method_name == "DynamicTuple_Demo" || command.method_name == "DynamicTuple_Basic" ||
        command.method_name == "DynamicTuple"|| command.method_name == "DynamicTuple_Dims") {
        ClassificationMainZcy(command, program_state, rules, traces, ans);
    } else if (command.method_name == "PSTSS" || command.method_name == "PartitionSort" ||
        command.method_name == "TupleMerge") {
        ClassificationMainPS(command, program_state, rules, traces, ans);
    } else {
        printf("No such method %s\n", command.method_name.c_str());
    }

    PrintProgramState(command, program_state);
    
    FreeRules(rules);
    FreeTraces(traces);
    return 0;
}