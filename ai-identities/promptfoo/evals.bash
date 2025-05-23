#!/bin/bash

usage() {
    echo "Usage: $0 create <template_flag> <new_name>. -c for conversation template, -f for followup template, -s for single prompts template"
    echo "       $0 evaluate <folder_1> <folder_1> ..."
    exit 1
}

SOURCE1="./sample_configs/sample_config_single"
SOURCE2="././sample_configs/sample_config_conversation"
SOURCE3="././sample_configs/sample_config_followup"
DESTINATION="./prompt_tests"
OUTPUT="./outputs"
COMMAND="$1"
shift

case "$COMMAND" in
    create)
        SOURCE_NAME="$1"
        NEW_NAME="$2"
        
        if [ "$SOURCE_NAME" == "-s" ]; then
            SOURCE="$SOURCE1"
        elif [ "$SOURCE_NAME" == "-c" ]; then
            SOURCE="$SOURCE2"
        elif [ "$SOURCE_NAME" == "-f" ]; then
            SOURCE="$SOURCE3"
        else
            echo "Error: Invalid option for test creation. Choose s, f or c."
            exit 1
        fi

        if [ -z "$NEW_NAME" ]; then
            echo "Missing folder name"
            exit 2
        fi

        if [ -d "$DESTINATION/$NEW_NAME" ]; then
            echo "Folder already exists"
            exit 3
        fi

        cp -R "$SOURCE" "$DESTINATION/$NEW_NAME"
        echo "Copied $SOURCE to $DESTINATION/$NEW_NAME"
        exit 0
        ;;
    evaluate)
        TARGET_FOLDER="$1"
        TIMESTAMP=$(date +%s)
        mkdir -p "./outputs/$TIMESTAMP"
        if [ -z "$TARGET_FOLDER" ]; then
            for folder in "$DESTINATION"/*/; do
                if [ -d "$folder" ]; then
                    echo $folder
                    folder_no_trailing=$(basename $folder)
                    (cd "$folder" && eval "npx promptfoo eval --no-cache --output ../../outputs/$TIMESTAMP/$folder_no_trailing.json")
                fi
            done
        else
            for folder in "$@"; do
                if [ -d "$DESTINATION/$folder/" ]; then
                    (cd "$DESTINATION/$folder/" && eval "npx promptfoo eval --no-cache  --output ../../outputs/$TIMESTAMP/$folder.json")
                fi
            done
        fi
        
        exit 0
        ;;
    *)
        usage
        ;;
esac
