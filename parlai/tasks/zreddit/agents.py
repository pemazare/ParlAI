# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import json
import logging
import os

from parlai.core.dialog_teacher import DialogTeacher


class DefaultTeacher(DialogTeacher):
    """Debugging task."""

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        opt['datafile'] = os.path.join(opt['datapath'], 'zreddit')
        self.id = 'zreddit'
        super().__init__(opt, shared)

    def setup_data(self, path):
        with open(path) as data_file:
            for line in data_file:
                try:
                    line_json = json.loads(line)
                    yield (line_json['text'], ), True
                except Exception:
                    logging.warning('data skipped: unable to parse from ' + path + ' : ' + line)
